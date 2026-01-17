# TASK-P4-008: TopicSynthesizer (Standalone Module)

```xml
<task_spec id="TASK-P4-008" version="2.0">
<metadata>
  <title>TopicSynthesizer Standalone Implementation</title>
  <status>COMPLETE</status>
  <layer>logic</layer>
  <sequence>34</sequence>
  <phase>4</phase>
  <implements>
    <requirement_ref>REQ-P4-04</requirement_ref>
    <requirement_ref>REQ-P4-05</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETE">TASK-P4-002</task_ref>
    <task_ref status="COMPLETE">TASK-P4-007</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
</metadata>

<critical_context>
## Current Codebase State (Audited 2026-01-17)

### CRITICAL: Existing Infrastructure - DO NOT RECREATE

**1. Topic synthesis ALREADY EXISTS in manager.rs (lines 481-534)**
The `MultiSpaceClusterManager::synthesize_topics()` method already has basic topic
synthesis. This task creates a **STANDALONE TopicSynthesizer struct** to decouple
synthesis logic from the manager for better testability and reusability.

**2. EmbedderCategory system EXISTS - `crates/context-graph-core/src/embeddings/category.rs`**
```rust
// DO NOT REDEFINE THESE - IMPORT FROM category.rs:
use crate::embeddings::category::{category_for, max_weighted_agreement, topic_threshold};

// category_for(embedder: Embedder) -> EmbedderCategory
// EmbedderCategory::topic_weight() -> f32
//   - Semantic: 1.0 (E1, E5, E6, E7, E10, E12, E13)
//   - Temporal: 0.0 (E2, E3, E4) - EXCLUDED per AP-60
//   - Relational: 0.5 (E8, E11)
//   - Structural: 0.5 (E9)
// max_weighted_agreement() -> 8.5
// topic_threshold() -> 2.5
```

**3. Topic types EXIST - `crates/context-graph-core/src/clustering/topic.rs`**
```rust
// ALREADY IMPLEMENTED:
pub struct TopicProfile { pub strengths: [f32; 13] }
impl TopicProfile {
    pub fn weighted_agreement(&self) -> f32;  // Uses category weights
    pub fn is_topic(&self) -> bool;           // >= topic_threshold()
    pub fn similarity(&self, other: &TopicProfile) -> f32;
}
pub struct Topic { id, name, profile, cluster_ids, member_memories, confidence, stability, created_at }
impl Topic {
    pub fn new(profile, cluster_ids, members) -> Self;
    pub fn is_valid(&self) -> bool;
    pub fn update_contributing_spaces(&mut self);
}
pub struct TopicStability { phase, age_hours, membership_churn, centroid_drift, ... }
pub enum TopicPhase { Emerging, Stable, Declining, Merging }
```

**4. ClusterMembership EXISTS - `crates/context-graph-core/src/clustering/membership.rs`**
```rust
pub struct ClusterMembership {
    pub memory_id: Uuid,
    pub space: Embedder,
    pub cluster_id: i32,          // -1 = NOISE_CLUSTER_ID
    pub membership_probability: f32,
    pub is_core_point: bool,
}
impl ClusterMembership {
    pub fn new(memory_id, space, cluster_id, probability, is_core) -> Self;
}
```

**5. Embedder enum EXISTS - `crates/context-graph-core/src/teleological/embedder.rs`**
```rust
pub enum Embedder {
    Semantic,           // E1  - index 0
    TemporalRecent,     // E2  - index 1
    TemporalPeriodic,   // E3  - index 2
    TemporalPositional, // E4  - index 3
    Causal,             // E5  - index 4
    Sparse,             // E6  - index 5
    Code,               // E7  - index 6
    Emotional,          // E8  - index 7
    Hdc,                // E9  - index 8
    Multimodal,         // E10 - index 9
    Entity,             // E11 - index 10
    LateInteraction,    // E12 - index 11
    KeywordSplade,      // E13 - index 12
}
impl Embedder {
    pub fn all() -> impl ExactSizeIterator<Item=Embedder>;
    pub fn index(self) -> usize;
}
```

### WRONG in Original Task Document

1. **Embedder variant names are WRONG** - Uses `Embedder::E1Semantic` but actual is `Embedder::Semantic`
2. **Redefines constants** - Task says define `TOPIC_THRESHOLD`, `MAX_WEIGHTED_AGREEMENT`, weight constants but these ALREADY EXIST in `category.rs`
3. **Uses `Embedder::E1Semantic` syntax** - Actual enum variants don't have "E1" prefix
4. **Test uses `count_shared_clusters`** - This function doesn't exist, should be `compute_weighted_agreement`
</critical_context>

<weighted_agreement_formula>
## Constitution-Mandated Formula (ARCH-09, AP-60)

```rust
// Between two memories A and B:
fn compute_weighted_agreement(A, B, mem_clusters) -> f32 {
    let mut weighted = 0.0f32;
    for embedder in Embedder::all() {
        let cluster_a = mem_clusters[A][embedder];
        let cluster_b = mem_clusters[B][embedder];

        // Both in same non-noise cluster
        if cluster_a != -1 && cluster_a == cluster_b {
            weighted += category_for(embedder).topic_weight();
        }
    }
    weighted
}

// Category weights (from category.rs - DO NOT REDEFINE):
// SEMANTIC (E1, E5, E6, E7, E10, E12, E13): 1.0 each = max 7.0
// TEMPORAL (E2, E3, E4): 0.0 each = max 0.0 (EXCLUDED per AP-60)
// RELATIONAL (E8, E11): 0.5 each = max 1.0
// STRUCTURAL (E9): 0.5 = max 0.5
// MAX_WEIGHTED_AGREEMENT = 8.5
// TOPIC_THRESHOLD = 2.5
```

### Examples from Constitution:
- 3 semantic spaces agreeing = 3.0 >= 2.5 -> TOPIC
- 2 semantic + 1 relational = 2.5 >= 2.5 -> TOPIC
- 2 semantic spaces only = 2.0 < 2.5 -> NOT TOPIC
- 5 temporal spaces = 0.0 < 2.5 -> NOT TOPIC (excluded per AP-60)
- 1 semantic + 3 relational = 2.5 >= 2.5 -> TOPIC
</weighted_agreement_formula>

<scope>
  <in_scope>
    - Create NEW file: crates/context-graph-core/src/clustering/synthesizer.rs
    - Implement TopicSynthesizer struct with configurable merge_threshold and min_silhouette
    - Implement synthesize_topics() taking HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;
    - Implement compute_weighted_agreement() using category_for().topic_weight()
    - Implement find_topic_mates() using Union-Find connected components
    - Implement merge_similar_topics() with configurable threshold (default 0.9)
    - Implement update_topic_stability() for membership changes
    - Add pub mod synthesizer to clustering/mod.rs
    - Export TopicSynthesizer from mod.rs
  </in_scope>
  <out_of_scope>
    - Redefining TOPIC_THRESHOLD, MAX_WEIGHTED_AGREEMENT (use category.rs)
    - Redefining embedder weights (use category_for().topic_weight())
    - Modifying existing manager.rs
    - Topic naming (future LLM integration)
    - Topic persistence
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-core/src/clustering/synthesizer.rs">
      //! Standalone topic synthesizer for cross-space topic discovery.
      //! Uses weighted agreement per ARCH-09, excluding temporal per AP-60.

      use std::collections::{HashMap, HashSet};
      use uuid::Uuid;
      use crate::embeddings::category::{category_for, max_weighted_agreement, topic_threshold};
      use crate::teleological::Embedder;
      use super::membership::ClusterMembership;
      use super::topic::{Topic, TopicProfile};
      use super::error::ClusterError;

      pub const DEFAULT_MERGE_THRESHOLD: f32 = 0.9;
      pub const DEFAULT_MIN_SILHOUETTE: f32 = 0.3;

      #[derive(Debug, Clone)]
      pub struct TopicSynthesizer {
          merge_similarity_threshold: f32,
          min_silhouette: f32,
      }

      impl Default for TopicSynthesizer { ... }

      impl TopicSynthesizer {
          pub fn new() -> Self;
          pub fn with_config(merge_threshold: f32, min_silhouette: f32) -> Self;

          pub fn synthesize_topics(
              &amp;self,
              memberships: &amp;HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;,
          ) -> Result&lt;Vec&lt;Topic&gt;, ClusterError&gt;;

          fn compute_weighted_agreement(
              &amp;self,
              mem_a: &amp;Uuid,
              mem_b: &amp;Uuid,
              mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;,
          ) -> f32;

          fn find_topic_mates(
              &amp;self,
              mem_clusters: &amp;HashMap&lt;Uuid, HashMap&lt;Embedder, i32&gt;&gt;,
          ) -> Vec&lt;Vec&lt;Uuid&gt;&gt;;

          pub fn update_topic_stability(
              &amp;self,
              topic: &amp;mut Topic,
              old_members: &amp;[Uuid],
              new_members: &amp;[Uuid],
          );
      }
    </signature>
  </signatures>

  <constraints>
    - MUST use category_for(embedder).topic_weight() for weights (DO NOT hardcode weights)
    - MUST use topic_threshold() from category.rs (2.5)
    - MUST use max_weighted_agreement() from category.rs (8.5)
    - Temporal embedders (E2-E4) MUST contribute 0.0 to weighted_agreement
    - Topics with fewer than 2 members MUST be filtered
    - Profile similarity uses TopicProfile::similarity() (cosine)
    - merge_similarity_threshold default = 0.9
  </constraints>
</definition_of_done>

<implementation>
## CORRECT Implementation (Use Actual Embedder Variant Names)

```rust
//! Standalone topic synthesizer for cross-space topic discovery.
//!
//! Uses weighted agreement formula per ARCH-09:
//! - SEMANTIC embedders (E1, E5, E6, E7, E10, E12, E13): 1.0 weight
//! - TEMPORAL embedders (E2, E3, E4): 0.0 weight (excluded per AP-60)
//! - RELATIONAL embedders (E8, E11): 0.5 weight
//! - STRUCTURAL embedder (E9): 0.5 weight

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::embeddings::category::{category_for, topic_threshold};
use crate::teleological::Embedder;

use super::error::ClusterError;
use super::membership::ClusterMembership;
use super::topic::{Topic, TopicProfile};

/// Default threshold for merging similar topics (profile cosine similarity).
pub const DEFAULT_MERGE_THRESHOLD: f32 = 0.9;

/// Default minimum silhouette score for valid clusters.
pub const DEFAULT_MIN_SILHOUETTE: f32 = 0.3;

/// Synthesizes topics from cross-space clustering using weighted agreement.
///
/// This is a standalone component that can be used independently of
/// `MultiSpaceClusterManager` for topic discovery.
#[derive(Debug, Clone)]
pub struct TopicSynthesizer {
    /// Threshold for merging similar topics (default 0.9).
    merge_similarity_threshold: f32,
    /// Minimum silhouette score for valid clusters (default 0.3).
    min_silhouette: f32,
}

impl Default for TopicSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicSynthesizer {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            merge_similarity_threshold: DEFAULT_MERGE_THRESHOLD,
            min_silhouette: DEFAULT_MIN_SILHOUETTE,
        }
    }

    /// Create with custom configuration.
    ///
    /// # Arguments
    /// * `merge_threshold` - Similarity threshold for merging topics (clamped 0.0..=1.0)
    /// * `min_silhouette` - Minimum silhouette score (clamped -1.0..=1.0)
    pub fn with_config(merge_threshold: f32, min_silhouette: f32) -> Self {
        Self {
            merge_similarity_threshold: merge_threshold.clamp(0.0, 1.0),
            min_silhouette: min_silhouette.clamp(-1.0, 1.0),
        }
    }

    /// Build map: memory_id -> (embedder -> cluster_id)
    fn build_mem_clusters_map(
        &self,
        memberships: &HashMap<Embedder, Vec<ClusterMembership>>,
    ) -> HashMap<Uuid, HashMap<Embedder, i32>> {
        let mut result: HashMap<Uuid, HashMap<Embedder, i32>> = HashMap::new();

        for (embedder, space_memberships) in memberships {
            for m in space_memberships {
                result
                    .entry(m.memory_id)
                    .or_default()
                    .insert(*embedder, m.cluster_id);
            }
        }

        result
    }

    /// Compute weighted agreement between two memories.
    ///
    /// Uses `category_for(embedder).topic_weight()` from category.rs.
    /// Temporal embedders (E2-E4) contribute 0.0 per AP-60.
    fn compute_weighted_agreement(
        &self,
        mem_a: &Uuid,
        mem_b: &Uuid,
        mem_clusters: &HashMap<Uuid, HashMap<Embedder, i32>>,
    ) -> f32 {
        let clusters_a = match mem_clusters.get(mem_a) {
            Some(c) => c,
            None => return 0.0,
        };
        let clusters_b = match mem_clusters.get(mem_b) {
            Some(c) => c,
            None => return 0.0,
        };

        let mut weighted = 0.0f32;
        for embedder in Embedder::all() {
            let ca = clusters_a.get(&embedder).copied().unwrap_or(-1);
            let cb = clusters_b.get(&embedder).copied().unwrap_or(-1);

            // Both in same non-noise cluster
            if ca != -1 && ca == cb {
                // Use category weight from category.rs (temporal = 0.0)
                weighted += category_for(embedder).topic_weight();
            }
        }
        weighted
    }

    /// Find groups of memories using Union-Find (connected components).
    ///
    /// An edge exists between memories if their weighted_agreement >= topic_threshold (2.5).
    fn find_topic_mates(
        &self,
        mem_clusters: &HashMap<Uuid, HashMap<Embedder, i32>>,
    ) -> Vec<Vec<Uuid>> {
        let memory_ids: Vec<Uuid> = mem_clusters.keys().cloned().collect();
        let n = memory_ids.len();

        if n == 0 {
            return Vec::new();
        }

        // Build edges for pairs meeting threshold
        let threshold = topic_threshold();
        let mut edges: Vec<(usize, usize)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let wa = self.compute_weighted_agreement(
                    &memory_ids[i],
                    &memory_ids[j],
                    mem_clusters,
                );
                if wa >= threshold {
                    edges.push((i, j));
                }
            }
        }

        // Union-Find with path compression
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        for (i, j) in edges {
            union(&mut parent, i, j);
        }

        // Group by component root
        let mut components: HashMap<usize, Vec<Uuid>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            components.entry(root).or_default().push(memory_ids[i]);
        }

        components.into_values().collect()
    }

    /// Compute topic profile from members.
    fn compute_topic_profile(
        &self,
        members: &[Uuid],
        mem_clusters: &HashMap<Uuid, HashMap<Embedder, i32>>,
    ) -> TopicProfile {
        let mut strengths = [0.0f32; 13];

        if members.is_empty() {
            return TopicProfile::new(strengths);
        }

        for embedder in Embedder::all() {
            let idx = embedder.index();
            let mut cluster_counts: HashMap<i32, usize> = HashMap::new();

            for mem_id in members {
                if let Some(clusters) = mem_clusters.get(mem_id) {
                    let cid = clusters.get(&embedder).copied().unwrap_or(-1);
                    if cid != -1 {
                        *cluster_counts.entry(cid).or_insert(0) += 1;
                    }
                }
            }

            // Strength = fraction in dominant cluster
            if let Some((_, &count)) = cluster_counts.iter().max_by_key(|(_, &c)| c) {
                strengths[idx] = count as f32 / members.len() as f32;
            }
        }

        TopicProfile::new(strengths)
    }

    /// Compute cluster_ids for topic (most common cluster per space).
    fn compute_cluster_ids(
        &self,
        members: &[Uuid],
        mem_clusters: &HashMap<Uuid, HashMap<Embedder, i32>>,
    ) -> HashMap<Embedder, i32> {
        let mut result = HashMap::new();

        for embedder in Embedder::all() {
            let mut counts: HashMap<i32, usize> = HashMap::new();

            for mem_id in members {
                if let Some(clusters) = mem_clusters.get(mem_id) {
                    let cid = clusters.get(&embedder).copied().unwrap_or(-1);
                    if cid != -1 {
                        *counts.entry(cid).or_insert(0) += 1;
                    }
                }
            }

            if let Some((&dominant, _)) = counts.iter().max_by_key(|(_, &c)| c) {
                result.insert(embedder, dominant);
            }
        }

        result
    }

    /// Merge highly similar topics.
    fn merge_similar_topics(&self, mut topics: Vec<Topic>) -> Vec<Topic> {
        if topics.len() <= 1 {
            return topics;
        }

        // Sort by member count descending
        topics.sort_by(|a, b| b.member_count().cmp(&a.member_count()));

        let mut merged: Vec<Topic> = Vec::new();
        let mut absorbed: HashSet<usize> = HashSet::new();

        for i in 0..topics.len() {
            if absorbed.contains(&i) {
                continue;
            }

            let mut current = topics[i].clone();

            for j in (i + 1)..topics.len() {
                if absorbed.contains(&j) {
                    continue;
                }

                let sim = current.profile.similarity(&topics[j].profile);
                if sim >= self.merge_similarity_threshold {
                    // Absorb j into current
                    for mem_id in &topics[j].member_memories {
                        if !current.member_memories.contains(mem_id) {
                            current.member_memories.push(*mem_id);
                        }
                    }
                    for (space, cid) in &topics[j].cluster_ids {
                        current.cluster_ids.entry(*space).or_insert(*cid);
                    }
                    absorbed.insert(j);
                }
            }

            current.update_contributing_spaces();
            merged.push(current);
        }

        merged
    }

    /// Main synthesis entry point.
    ///
    /// Discovers topics from cluster memberships where memory pairs have
    /// weighted_agreement >= 2.5 (topic_threshold).
    pub fn synthesize_topics(
        &self,
        memberships: &HashMap<Embedder, Vec<ClusterMembership>>,
    ) -> Result<Vec<Topic>, ClusterError> {
        let mem_clusters = self.build_mem_clusters_map(memberships);

        if mem_clusters.is_empty() {
            return Ok(Vec::new());
        }

        // Find connected components
        let groups = self.find_topic_mates(&mem_clusters);

        // Create topics from groups with >= 2 members
        let mut topics: Vec<Topic> = groups
            .into_iter()
            .filter(|g| g.len() >= 2)
            .map(|members| {
                let profile = self.compute_topic_profile(&members, &mem_clusters);
                let cluster_ids = self.compute_cluster_ids(&members, &mem_clusters);
                Topic::new(profile, cluster_ids, members)
            })
            .filter(|t| t.is_valid())
            .collect();

        // Merge similar topics
        topics = self.merge_similar_topics(topics);

        Ok(topics)
    }

    /// Update topic stability based on membership changes.
    pub fn update_topic_stability(
        &self,
        topic: &mut Topic,
        old_members: &[Uuid],
        new_members: &[Uuid],
    ) {
        let old_set: HashSet<_> = old_members.iter().collect();
        let new_set: HashSet<_> = new_members.iter().collect();

        let sym_diff = old_set.symmetric_difference(&new_set).count();
        let union_size = old_set.union(&new_set).count();

        let churn = if union_size > 0 {
            sym_diff as f32 / union_size as f32
        } else {
            0.0
        };

        topic.stability.membership_churn = churn;

        let age = chrono::Utc::now() - topic.created_at;
        topic.stability.age_hours = age.num_minutes() as f32 / 60.0;

        topic.stability.update_phase();
    }
}
```

## CORRECT Tests (Use Actual Embedder Variant Names)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::category::topic_threshold;

    /// Create memberships where two memories share clusters in specified spaces.
    fn create_shared_memberships(
        id1: Uuid,
        id2: Uuid,
        shared_spaces: &[Embedder],
    ) -> HashMap<Embedder, Vec<ClusterMembership>> {
        let mut memberships = HashMap::new();

        for embedder in Embedder::all() {
            let cluster_id = if shared_spaces.contains(&embedder) { 1 } else { -1 };
            let other_cluster_id = if shared_spaces.contains(&embedder) { 1 } else { 99 };

            memberships.entry(embedder).or_insert_with(Vec::new).push(
                ClusterMembership::new(id1, embedder, cluster_id, 0.9, true)
            );
            memberships.entry(embedder).or_insert_with(Vec::new).push(
                ClusterMembership::new(id2, embedder, other_cluster_id, 0.9, true)
            );
        }

        memberships
    }

    #[test]
    fn test_weighted_agreement_3_semantic_forms_topic() {
        // 3 semantic spaces = 3.0 >= 2.5 = TOPIC
        let synthesizer = TopicSynthesizer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // Semantic, Causal, Code are all SEMANTIC category (weight 1.0)
        let memberships = create_shared_memberships(
            id1, id2,
            &[Embedder::Semantic, Embedder::Causal, Embedder::Code]
        );

        let result = synthesizer.synthesize_topics(&memberships).unwrap();

        println!("STATE: 3 semantic spaces (E1, E5, E7) sharing cluster 1");
        println!("RESULT: {} topic(s) formed", result.len());

        assert_eq!(result.len(), 1, "3 semantic spaces (3.0) should form 1 topic");
        println!("[PASS] 3 semantic spaces agreeing = 3.0 -> TOPIC");
    }

    #[test]
    fn test_weighted_agreement_2_semantic_1_relational_forms_topic() {
        // 2 semantic + 1 relational = 2.0 + 0.5 = 2.5 = TOPIC
        let synthesizer = TopicSynthesizer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        // Semantic=1.0, Causal=1.0, Emotional=0.5 (relational)
        let memberships = create_shared_memberships(
            id1, id2,
            &[Embedder::Semantic, Embedder::Causal, Embedder::Emotional]
        );

        let result = synthesizer.synthesize_topics(&memberships).unwrap();

        println!("STATE: 2 semantic + 1 relational = 2.5");
        println!("RESULT: {} topic(s) formed", result.len());

        assert_eq!(result.len(), 1, "2 semantic + 1 relational (2.5) should form topic");
        println!("[PASS] 2 semantic + 1 relational = 2.5 -> TOPIC");
    }

    #[test]
    fn test_weighted_agreement_2_semantic_only_no_topic() {
        // 2 semantic only = 2.0 < 2.5 = NO TOPIC
        let synthesizer = TopicSynthesizer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let memberships = create_shared_memberships(
            id1, id2,
            &[Embedder::Semantic, Embedder::Causal] // 1.0 + 1.0 = 2.0 < 2.5
        );

        let result = synthesizer.synthesize_topics(&memberships).unwrap();

        println!("STATE: 2 semantic only = 2.0");
        println!("RESULT: {} topic(s) formed", result.len());

        assert!(result.is_empty(), "2 semantic only (2.0) should NOT form topic");
        println!("[PASS] 2 semantic spaces only = 2.0 -> NOT TOPIC");
    }

    #[test]
    fn test_temporal_excluded_from_agreement() {
        // All temporal = 0.0 = NO TOPIC (AP-60)
        let synthesizer = TopicSynthesizer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let memberships = create_shared_memberships(
            id1, id2,
            &[Embedder::TemporalRecent, Embedder::TemporalPeriodic, Embedder::TemporalPositional]
        );

        // Manually compute to verify
        let mem_clusters = synthesizer.build_mem_clusters_map(&memberships);
        let wa = synthesizer.compute_weighted_agreement(&id1, &id2, &mem_clusters);

        println!("STATE: 3 temporal spaces sharing clusters");
        println!("weighted_agreement = {} (should be 0.0)", wa);

        assert_eq!(wa, 0.0, "Temporal-only agreement should be 0.0");

        let result = synthesizer.synthesize_topics(&memberships).unwrap();
        println!("RESULT: {} topic(s) formed", result.len());

        assert!(result.is_empty(), "Temporal-only should NOT form topic");
        println!("[PASS] All temporal spaces = 0.0 -> NOT TOPIC (AP-60 verified)");
    }

    #[test]
    fn test_empty_input() {
        let synthesizer = TopicSynthesizer::new();
        let memberships: HashMap<Embedder, Vec<ClusterMembership>> = HashMap::new();

        println!("STATE BEFORE: empty memberships");
        let result = synthesizer.synthesize_topics(&memberships).unwrap();
        println!("STATE AFTER: {} topics", result.len());

        assert!(result.is_empty());
        println!("[PASS] Empty input -> empty output");
    }

    #[test]
    fn test_single_memory_no_topic() {
        let synthesizer = TopicSynthesizer::new();
        let id = Uuid::new_v4();

        let mut memberships = HashMap::new();
        memberships.insert(Embedder::Semantic, vec![
            ClusterMembership::new(id, Embedder::Semantic, 1, 0.9, true)
        ]);

        println!("STATE BEFORE: 1 memory only");
        let result = synthesizer.synthesize_topics(&memberships).unwrap();
        println!("STATE AFTER: {} topics", result.len());

        assert!(result.is_empty(), "Cannot form topic with single memory");
        println!("[PASS] Single memory -> no topic");
    }

    #[test]
    fn test_update_stability_churn() {
        let synthesizer = TopicSynthesizer::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let profile = TopicProfile::new([0.8; 13]);
        let mut topic = Topic::new(profile, HashMap::new(), vec![id1, id2]);

        let old_members = vec![id1, id2];
        let new_members = vec![id1, id3]; // id2 left, id3 joined

        println!("STATE BEFORE: members = [id1, id2], churn = {}", topic.stability.membership_churn);
        synthesizer.update_topic_stability(&mut topic, &old_members, &new_members);
        println!("STATE AFTER: churn = {}", topic.stability.membership_churn);

        // churn = 2 changes / 3 total = 0.666...
        assert!(topic.stability.membership_churn > 0.5, "Churn should be > 0.5");
        println!("[PASS] Stability churn computed: {:.3}", topic.stability.membership_churn);
    }
}
```
</implementation>

<files_to_create>
  <file path="crates/context-graph-core/src/clustering/synthesizer.rs">TopicSynthesizer implementation</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-core/src/clustering/mod.rs">
    Add line: pub mod synthesizer;
    Add to exports: pub use synthesizer::{TopicSynthesizer, DEFAULT_MERGE_THRESHOLD, DEFAULT_MIN_SILHOUETTE};
  </file>
</files_to_modify>

<full_state_verification>
## MANDATORY: Full State Verification Protocol

After completing the implementation, you MUST verify using these steps.
DO NOT rely on return values alone.

### 1. Source of Truth
- **Input**: HashMap&lt;Embedder, Vec&lt;ClusterMembership&gt;&gt;
- **Output**: Vec&lt;Topic&gt; with confidence computed as weighted_agreement/8.5

### 2. Execute & Inspect
```bash
# Compile first
cargo check --package context-graph-core

# Run synthesizer tests with output
cargo test --package context-graph-core synthesizer -- --nocapture

# Run all clustering tests
cargo test --package context-graph-core clustering -- --nocapture
```

### 3. Boundary & Edge Case Verification (MUST print before/after state)
Each test case MUST:
1. Print state BEFORE operation
2. Execute operation
3. Print state AFTER operation
4. Assert expected outcome

### 4. Evidence of Success
Test output MUST show:
- `weighted_agreement` computed correctly (temporal = 0.0)
- Topics only formed when weighted_agreement >= 2.5
- All `[PASS]` markers for each test case
</full_state_verification>

<validation_criteria>
  <criterion>File synthesizer.rs compiles without errors</criterion>
  <criterion>TopicSynthesizer::new() creates with default thresholds</criterion>
  <criterion>compute_weighted_agreement uses category_for().topic_weight()</criterion>
  <criterion>Temporal embedders (E2-E4) contribute 0.0 to weighted_agreement</criterion>
  <criterion>Topics formed ONLY when weighted_agreement >= 2.5 (ARCH-09)</criterion>
  <criterion>topic.confidence = weighted_agreement / 8.5</criterion>
  <criterion>Similar topics merged when similarity >= 0.9</criterion>
  <criterion>Stability churn computed correctly on membership change</criterion>
  <criterion>All edge case tests pass with printed state before/after</criterion>
</validation_criteria>

<test_commands>
  <command description="Check compilation">cargo check --package context-graph-core</command>
  <command description="Run synthesizer tests">cargo test --package context-graph-core synthesizer -- --nocapture</command>
  <command description="Run all clustering tests">cargo test --package context-graph-core clustering -- --nocapture</command>
</test_commands>
</task_spec>
```

## Execution Checklist

- [x] Read existing files first:
  - [x] `crates/context-graph-core/src/embeddings/category.rs` - verify category_for(), topic_threshold(), max_weighted_agreement()
  - [x] `crates/context-graph-core/src/clustering/topic.rs` - verify Topic, TopicProfile types
  - [x] `crates/context-graph-core/src/clustering/membership.rs` - verify ClusterMembership
  - [x] `crates/context-graph-core/src/teleological/embedder.rs` - verify Embedder enum variants
- [x] Create `crates/context-graph-core/src/clustering/synthesizer.rs`
- [x] Import from category.rs (DO NOT redefine constants)
- [x] Implement TopicSynthesizer struct
- [x] Implement build_mem_clusters_map()
- [x] Implement compute_weighted_agreement() using category_for().topic_weight()
- [x] Implement find_topic_mates() with Union-Find
- [x] Implement compute_topic_profile()
- [x] Implement compute_cluster_ids()
- [x] Implement merge_similar_topics()
- [x] Implement synthesize_topics()
- [x] Implement update_topic_stability()
- [x] Add `pub mod synthesizer` to mod.rs
- [x] Add re-exports to mod.rs
- [x] Write all test cases with correct Embedder variants (Semantic, not E1Semantic)
- [x] `cargo check --package context-graph-core`
- [x] `cargo test --package context-graph-core synthesizer -- --nocapture`
- [x] **VERIFY**: All tests print state before/after
- [x] **VERIFY**: Temporal contributes 0.0 to weighted_agreement
- [x] **VERIFY**: Topics only form when weighted_agreement >= 2.5
- [ ] Proceed to TASK-P4-009

## Completion Summary (2026-01-17)

### Files Created
- `crates/context-graph-core/src/clustering/synthesizer.rs` (475 lines)
- `crates/context-graph-core/tests/synthesizer_edge_case_test.rs` (250 lines)

### Files Modified
- `crates/context-graph-core/src/clustering/mod.rs` - Added synthesizer module and exports

### Test Results
- 18 unit tests PASSED
- 13 edge case integration tests PASSED
- 243 total clustering tests PASSED

### Verification Evidence
All edge cases verified with state logging:
1. Empty input → empty output
2. Single memory → no topic
3. All noise → no topic
4. Threshold boundary (exactly 2.5) → topic formed
5. Below threshold (2.0) → no topic
6. Temporal only → 0.0 contribution (AP-60 verified)
7. Max weighted agreement → 8.5
8. Different clusters → no agreement
9. Mixed temporal + semantic → temporal adds 0.0
10. Multiple topic groups → separate topics

### Code Simplification Applied
- Used `let-else` pattern for early returns
- Extracted shared logic into `count_clusters_for_space` helper
- Used `sort_by_key` with `Reverse` for idiomatic sorting

## Key Corrections from Original Task

| Original (WRONG) | Corrected |
|------------------|-----------|
| `Embedder::E1Semantic` | `Embedder::Semantic` |
| `Embedder::E5Causal` | `Embedder::Causal` |
| `Embedder::E7Code` | `Embedder::Code` |
| `Embedder::E2TempRecent` | `Embedder::TemporalRecent` |
| `Embedder::E8Emotional` | `Embedder::Emotional` |
| `Embedder::E9HDC` | `Embedder::Hdc` |
| Redefine `TOPIC_THRESHOLD` | Use `topic_threshold()` from category.rs |
| Redefine weight constants | Use `category_for(e).topic_weight()` |
| `count_shared_clusters` function | Does not exist - use `compute_weighted_agreement` |
