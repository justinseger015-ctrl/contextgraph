# Module 5: UTL Integration - Functional Specification

**Version**: 1.0.0
**Phase**: 4
**Duration**: 4 weeks
**Dependencies**: Module 4 (Knowledge Graph)
**Status**: Specification Complete

---

## 1. Executive Summary

Module 5 implements the Unified Theory of Learning (UTL), the core learning equation that governs how the Context Graph acquires, prioritizes, and strengthens memories. UTL bridges the embedding pipeline (Module 3) and knowledge graph (Module 4) with biologically-inspired learning dynamics that determine what gets remembered, how important it is, and when consolidation occurs.

### 1.1 Key Deliverables

1. **UTLProcessor** - Core computation engine for learning signals
2. **Surprise Computation** - KL divergence-based novelty detection
3. **Coherence Tracking** - Rolling window graph-based consistency
4. **Emotional Weighting** - Valence/arousal modulation with decay
5. **Phase Oscillator** - Consolidation state management
6. **Johari Quadrant Classifier** - Memory categorization system
7. **Lifecycle State Machine** - Cold-start to maturity transitions

---

## 2. UTL Core Equation

### 2.1 Primary Formula

```
L = f((delta_s * delta_c) * w_e * cos(phi))

Where:
- L: Learning magnitude (output) in [0, 1]
- delta_s: Entropy change (surprise/novelty) in [0, 1]
- delta_c: Coherence change (understanding) in [0, 1]
- w_e: Emotional modulation weight in [0.5, 1.5]
- phi: Phase synchronization angle in [0, pi]
- f: Sigmoid activation function for normalization
```

### 2.2 Mathematical Formulation

```rust
/// Compute learning magnitude from UTL components
///
/// # Arguments
/// * `delta_s` - Surprise/entropy change, clamped to [0, 1]
/// * `delta_c` - Coherence change, clamped to [0, 1]
/// * `w_e` - Emotional weight, clamped to [0.5, 1.5]
/// * `phi` - Phase angle, clamped to [0, PI]
///
/// # Returns
/// Learning magnitude L in [0, 1]
///
/// # Constraint: Operation_Latency < 100us
pub fn compute_learning_magnitude(
    delta_s: f32,
    delta_c: f32,
    w_e: f32,
    phi: f32,
) -> f32 {
    // Clamp inputs to valid ranges (prevents NaN/Infinity)
    let delta_s = delta_s.clamp(0.0, 1.0);
    let delta_c = delta_c.clamp(0.0, 1.0);
    let w_e = w_e.clamp(0.5, 1.5);
    let phi = phi.clamp(0.0, std::f32::consts::PI);

    // Core UTL computation
    let raw_signal = (delta_s * delta_c) * w_e * phi.cos();

    // Sigmoid normalization to [0, 1]
    sigmoid(raw_signal * LEARNING_SCALE_FACTOR)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

### 2.3 Loss Function

```
J = lambda_task * L_task + lambda_semantic * L_semantic + lambda_dyn * (1 - L)

Where:
- lambda_task = 0.4 (task-specific loss weight)
- lambda_semantic = 0.3 (semantic similarity loss weight)
- lambda_dyn = 0.3 (dynamic learning loss weight)
```

### 2.4 Configuration Parameters

```yaml
# config/utl.yaml
utl:
  # Core equation parameters
  learning_scale_factor: 2.0  # Scales raw signal before sigmoid

  # Weight bounds
  weights:
    lambda_task: 0.4
    lambda_semantic: 0.3
    lambda_dyn: 0.3

  # Input ranges (for validation and clamping)
  ranges:
    delta_s:
      min: 0.0
      max: 1.0
    delta_c:
      min: 0.0
      max: 1.0
    w_e:
      min: 0.5
      max: 1.5
    phi:
      min: 0.0
      max: 3.14159265  # PI

  # Thresholds
  thresholds:
    consolidation_trigger: 0.7  # L > this triggers consolidation
    salience_update_min: 0.1    # Minimum L to update salience
    surprise_significant: 0.6   # delta_s > this is "surprising"
```

---

## 3. Surprise Computation (delta_s)

### 3.1 Overview

Surprise measures how unexpected the input is relative to the current context. High surprise indicates novel or unexpected information.

### 3.2 Computation Methods

#### 3.2.1 KL Divergence Method (Primary)

```rust
/// Compute surprise via KL divergence between predicted and observed distributions
///
/// # Formula
/// delta_s = KL(P_observed || P_predicted) / max_kl
///
/// Where max_kl normalizes to [0, 1]
pub fn compute_surprise_kl(
    observed_embedding: &[f32; 1536],
    context_embeddings: &[&[f32; 1536]],
    config: &SurpriseConfig,
) -> f32 {
    if context_embeddings.is_empty() {
        // Maximum surprise when no context
        return config.max_surprise_no_context;
    }

    // Compute predicted distribution (mean of context)
    let predicted = compute_centroid(context_embeddings);

    // Softmax normalization for probability distributions
    let p_obs = softmax_normalize(observed_embedding);
    let p_pred = softmax_normalize(&predicted);

    // KL divergence with numerical stability
    let kl_div = kl_divergence(&p_obs, &p_pred, config.epsilon);

    // Normalize to [0, 1]
    (kl_div / config.max_kl_value).clamp(0.0, 1.0)
}

fn kl_divergence(p: &[f32], q: &[f32], epsilon: f32) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            let qi_safe = qi.max(epsilon);
            let pi_safe = pi.max(epsilon);
            pi_safe * (pi_safe / qi_safe).ln()
        })
        .sum::<f32>()
        .max(0.0)  // KL is non-negative
}
```

#### 3.2.2 Embedding Distance Method (Secondary)

```rust
/// Compute surprise via cosine distance from context centroid
///
/// Lower similarity = higher surprise
pub fn compute_surprise_distance(
    observed_embedding: &[f32; 1536],
    context_embeddings: &[&[f32; 1536]],
    config: &SurpriseConfig,
) -> f32 {
    if context_embeddings.is_empty() {
        return config.max_surprise_no_context;
    }

    let centroid = compute_centroid(context_embeddings);
    let similarity = cosine_similarity(observed_embedding, &centroid);

    // Convert similarity [-1, 1] to surprise [0, 1]
    // similarity 1.0 -> surprise 0.0
    // similarity 0.0 -> surprise 0.5
    // similarity -1.0 -> surprise 1.0
    ((1.0 - similarity) / 2.0).clamp(0.0, 1.0)
}
```

#### 3.2.3 Ensemble Surprise

```rust
/// Combine multiple surprise methods with configurable weights
pub fn compute_surprise_ensemble(
    observed: &[f32; 1536],
    context: &[&[f32; 1536]],
    config: &SurpriseConfig,
) -> f32 {
    let kl_surprise = compute_surprise_kl(observed, context, config);
    let dist_surprise = compute_surprise_distance(observed, context, config);

    // Weighted combination
    let surprise = config.kl_weight * kl_surprise
                 + config.distance_weight * dist_surprise;

    surprise.clamp(0.0, 1.0)
}
```

### 3.3 Configuration

```yaml
surprise:
  # Method weights (must sum to 1.0)
  kl_weight: 0.6
  distance_weight: 0.4

  # KL divergence parameters
  kl:
    epsilon: 1e-10          # Numerical stability
    max_kl_value: 10.0      # Normalization ceiling
    temperature: 1.0        # Softmax temperature

  # Context window
  context_window_size: 50   # Recent nodes for context
  context_decay: 0.95       # Exponential decay for older context

  # Edge cases
  max_surprise_no_context: 0.9  # Surprise when context is empty
  min_context_for_kl: 3         # Minimum nodes for KL method
```

---

## 4. Coherence Tracking (delta_c)

### 4.1 Overview

Coherence measures how well the new information integrates with existing knowledge. High coherence indicates the information fits well with the current understanding.

### 4.2 Rolling Window Coherence

```rust
/// Track coherence using a rolling window of recent memories
pub struct CoherenceTracker {
    /// Circular buffer of recent node embeddings
    window: VecDeque<CoherenceEntry>,
    /// Maximum window size
    max_size: usize,
    /// Graph reference for structural coherence
    graph: Arc<RwLock<KnowledgeGraph>>,
}

pub struct CoherenceEntry {
    pub node_id: Uuid,
    pub embedding: [f32; 1536],
    pub timestamp: DateTime<Utc>,
    pub importance: f32,
}

impl CoherenceTracker {
    /// Compute coherence change for new input
    ///
    /// # Returns
    /// delta_c in [0, 1] where:
    /// - 1.0 = Perfectly coherent with context
    /// - 0.0 = Completely incoherent
    pub fn compute_coherence(
        &self,
        new_embedding: &[f32; 1536],
        new_content: &str,
        config: &CoherenceConfig,
    ) -> f32 {
        let semantic_coherence = self.compute_semantic_coherence(new_embedding, config);
        let structural_coherence = self.compute_structural_coherence(new_content, config);
        let contradiction_penalty = self.compute_contradiction_penalty(new_embedding, config);

        // Weighted combination with penalty
        let raw_coherence = config.semantic_weight * semantic_coherence
                         + config.structural_weight * structural_coherence;

        // Apply contradiction penalty
        (raw_coherence * (1.0 - contradiction_penalty)).clamp(0.0, 1.0)
    }

    /// Semantic coherence via average similarity to window
    fn compute_semantic_coherence(
        &self,
        embedding: &[f32; 1536],
        config: &CoherenceConfig,
    ) -> f32 {
        if self.window.is_empty() {
            return config.default_coherence_empty;
        }

        let similarities: Vec<f32> = self.window
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let sim = cosine_similarity(embedding, &entry.embedding);
                // Apply recency decay
                let decay = config.recency_decay.powi((self.window.len() - i) as i32);
                // Apply importance weighting
                sim * decay * entry.importance
            })
            .collect();

        let sum: f32 = similarities.iter().sum();
        let weight_sum: f32 = self.window
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let decay = config.recency_decay.powi((self.window.len() - i) as i32);
                decay * entry.importance
            })
            .sum();

        if weight_sum > 0.0 {
            (sum / weight_sum).clamp(0.0, 1.0)
        } else {
            config.default_coherence_empty
        }
    }

    /// Structural coherence via graph connectivity
    fn compute_structural_coherence(
        &self,
        content: &str,
        config: &CoherenceConfig,
    ) -> f32 {
        // Check how many existing nodes the new content can connect to
        let graph = self.graph.read().unwrap();

        // Extract entities/concepts from content
        let concepts = extract_concepts(content);

        if concepts.is_empty() {
            return config.default_coherence_no_concepts;
        }

        let mut connection_score = 0.0;
        for concept in &concepts {
            if let Some(node) = graph.find_by_concept(concept) {
                // Check edge density around the node
                let edge_count = graph.edge_count(node.id);
                let normalized_edges = (edge_count as f32 / config.max_edge_normalization).min(1.0);
                connection_score += normalized_edges;
            }
        }

        (connection_score / concepts.len() as f32).clamp(0.0, 1.0)
    }

    /// Detect contradictions with existing knowledge
    fn compute_contradiction_penalty(
        &self,
        embedding: &[f32; 1536],
        config: &CoherenceConfig,
    ) -> f32 {
        let graph = self.graph.read().unwrap();

        // Find highly similar nodes
        let similar_nodes = graph.search_similar(embedding, config.contradiction_search_k);

        let mut max_contradiction = 0.0;
        for (node, similarity) in similar_nodes {
            if similarity > config.contradiction_similarity_threshold {
                // Check for semantic contradiction via edge analysis
                if let Some(contradiction_edge) = graph.find_contradiction_edge(node.id) {
                    let contradiction_strength = contradiction_edge.weight * similarity;
                    max_contradiction = max_contradiction.max(contradiction_strength);
                }
            }
        }

        max_contradiction.clamp(0.0, config.max_contradiction_penalty)
    }

    /// Update window with new entry
    pub fn update(&mut self, entry: CoherenceEntry) {
        if self.window.len() >= self.max_size {
            self.window.pop_front();
        }
        self.window.push_back(entry);
    }
}
```

### 4.3 Configuration

```yaml
coherence:
  # Window parameters
  window_size: 100          # Maximum entries in rolling window
  recency_decay: 0.98       # Decay factor per position

  # Weight distribution
  semantic_weight: 0.6      # Weight for embedding similarity
  structural_weight: 0.4    # Weight for graph connectivity

  # Contradiction detection
  contradiction_search_k: 20
  contradiction_similarity_threshold: 0.85
  max_contradiction_penalty: 0.5

  # Defaults for edge cases
  default_coherence_empty: 0.5       # When window is empty
  default_coherence_no_concepts: 0.4 # When no concepts extracted

  # Structural
  max_edge_normalization: 50.0  # Edges for max structural score
```

---

## 5. Emotional Weighting (w_e)

### 5.1 Overview

Emotional weighting modulates learning based on the emotional significance of content. Content with high emotional valence (positive or negative) and arousal receives stronger learning signals.

### 5.2 Implementation

```rust
/// Emotional state for learning modulation
pub struct EmotionalState {
    /// Valence: positive (1.0) to negative (-1.0)
    pub valence: f32,
    /// Arousal: calm (0.0) to excited (1.0)
    pub arousal: f32,
    /// Timestamp for decay calculation
    pub timestamp: DateTime<Utc>,
}

pub struct EmotionalWeightCalculator {
    /// Current emotional state
    current_state: EmotionalState,
    /// Decay rate per second
    decay_rate: f32,
    /// Baseline weight when no emotion
    baseline_weight: f32,
}

impl EmotionalWeightCalculator {
    /// Compute emotional weight w_e from content
    ///
    /// # Returns
    /// w_e in [0.5, 1.5]
    pub fn compute_weight(
        &mut self,
        content: &str,
        config: &EmotionalConfig,
    ) -> f32 {
        // Extract emotional signals from content
        let (valence, arousal) = self.extract_emotion(content, config);

        // Update state with decay
        self.update_state(valence, arousal, config);

        // Compute weight from state
        self.state_to_weight(config)
    }

    /// Extract valence and arousal from content
    fn extract_emotion(&self, content: &str, config: &EmotionalConfig) -> (f32, f32) {
        // Method 1: Lexicon-based sentiment
        let lexicon_valence = self.lexicon_sentiment(content, config);

        // Method 2: Punctuation/capitalization heuristics
        let arousal_heuristic = self.arousal_heuristics(content, config);

        // Method 3: Keyword detection
        let keyword_boost = self.emotional_keywords(content, config);

        let valence = (lexicon_valence + keyword_boost.0).clamp(-1.0, 1.0);
        let arousal = (arousal_heuristic + keyword_boost.1).clamp(0.0, 1.0);

        (valence, arousal)
    }

    /// Update emotional state with decay
    fn update_state(&mut self, new_valence: f32, new_arousal: f32, config: &EmotionalConfig) {
        let now = Utc::now();
        let elapsed = (now - self.current_state.timestamp).num_milliseconds() as f32 / 1000.0;

        // Exponential decay
        let decay_factor = (-self.decay_rate * elapsed).exp();

        // Blend decayed state with new input
        self.current_state.valence = decay_factor * self.current_state.valence
                                   + (1.0 - decay_factor) * new_valence;
        self.current_state.arousal = decay_factor * self.current_state.arousal
                                   + (1.0 - decay_factor) * new_arousal;
        self.current_state.timestamp = now;

        // Clamp to valid ranges
        self.current_state.valence = self.current_state.valence.clamp(-1.0, 1.0);
        self.current_state.arousal = self.current_state.arousal.clamp(0.0, 1.0);
    }

    /// Convert emotional state to weight
    fn state_to_weight(&self, config: &EmotionalConfig) -> f32 {
        // Absolute valence matters (strong emotions boost learning)
        let valence_magnitude = self.current_state.valence.abs();

        // Combined emotional intensity
        let intensity = config.valence_weight * valence_magnitude
                      + config.arousal_weight * self.current_state.arousal;

        // Map intensity [0, 1] to weight [0.5, 1.5]
        let weight = self.baseline_weight + intensity * config.intensity_scale;

        weight.clamp(0.5, 1.5)
    }

    /// Lexicon-based sentiment analysis
    fn lexicon_sentiment(&self, content: &str, config: &EmotionalConfig) -> f32 {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        let mut sentiment_sum = 0.0;
        let mut word_count = 0;

        for word in &words {
            let normalized = word.to_lowercase();
            if let Some(score) = config.sentiment_lexicon.get(&normalized) {
                sentiment_sum += score;
                word_count += 1;
            }
        }

        if word_count > 0 {
            sentiment_sum / word_count as f32
        } else {
            0.0
        }
    }

    /// Arousal heuristics from punctuation/formatting
    fn arousal_heuristics(&self, content: &str, config: &EmotionalConfig) -> f32 {
        let exclamation_count = content.matches('!').count() as f32;
        let question_count = content.matches('?').count() as f32;
        let caps_ratio = content.chars().filter(|c| c.is_uppercase()).count() as f32
                       / content.len().max(1) as f32;

        let arousal = config.exclamation_weight * (exclamation_count / 10.0).min(1.0)
                    + config.question_weight * (question_count / 5.0).min(1.0)
                    + config.caps_weight * caps_ratio;

        arousal.clamp(0.0, 1.0)
    }

    /// Detect emotional keywords
    fn emotional_keywords(&self, content: &str, config: &EmotionalConfig) -> (f32, f32) {
        let content_lower = content.to_lowercase();

        let mut valence_boost = 0.0;
        let mut arousal_boost = 0.0;

        for (keyword, (v, a)) in &config.emotional_keywords {
            if content_lower.contains(keyword) {
                valence_boost += v;
                arousal_boost += a;
            }
        }

        (valence_boost.clamp(-0.5, 0.5), arousal_boost.clamp(0.0, 0.5))
    }
}
```

### 5.3 Configuration

```yaml
emotional:
  # Decay parameters
  decay_rate: 0.1           # Decay per second
  baseline_weight: 1.0      # Weight when neutral

  # Component weights
  valence_weight: 0.6       # Weight of valence in intensity
  arousal_weight: 0.4       # Weight of arousal in intensity
  intensity_scale: 0.5      # Scale factor for intensity -> weight

  # Heuristic weights
  exclamation_weight: 0.3
  question_weight: 0.2
  caps_weight: 0.2

  # Output range
  weight_min: 0.5
  weight_max: 1.5

  # Keyword examples (expanded in full config)
  emotional_keywords:
    "urgent": [0.0, 0.8]
    "critical": [0.0, 0.9]
    "error": [-0.3, 0.6]
    "success": [0.7, 0.5]
    "failed": [-0.5, 0.4]
    "important": [0.1, 0.5]
    "danger": [-0.4, 0.9]
    "excellent": [0.8, 0.4]
```

---

## 6. Phase Oscillator (phi)

### 6.1 Overview

The phase oscillator tracks the consolidation state of the memory system. It cycles between encoding (phi near 0) and consolidation (phi near PI) phases, inspired by biological theta rhythms.

### 6.2 Implementation

```rust
/// Phase oscillator for memory consolidation timing
pub struct PhaseOscillator {
    /// Current phase angle in [0, PI]
    phi: f32,
    /// Oscillation frequency (cycles per second)
    frequency: f32,
    /// Last update timestamp
    last_update: Instant,
    /// External modulation factor
    modulation: f32,
}

pub enum ConsolidationPhase {
    /// Encoding: optimal for new memories (phi near 0)
    Encoding,
    /// Transition: between states
    Transition,
    /// Consolidation: optimal for strengthening (phi near PI)
    Consolidation,
}

impl PhaseOscillator {
    pub fn new(config: &PhaseConfig) -> Self {
        Self {
            phi: 0.0,
            frequency: config.base_frequency,
            last_update: Instant::now(),
            modulation: 1.0,
        }
    }

    /// Get current phase angle
    ///
    /// # Returns
    /// phi in [0, PI]
    pub fn get_phi(&mut self) -> f32 {
        self.update();
        self.phi
    }

    /// Get current consolidation phase
    pub fn get_phase(&mut self) -> ConsolidationPhase {
        self.update();

        // Phase boundaries (configurable)
        const ENCODING_THRESHOLD: f32 = std::f32::consts::PI / 3.0;
        const CONSOLIDATION_THRESHOLD: f32 = 2.0 * std::f32::consts::PI / 3.0;

        if self.phi < ENCODING_THRESHOLD {
            ConsolidationPhase::Encoding
        } else if self.phi > CONSOLIDATION_THRESHOLD {
            ConsolidationPhase::Consolidation
        } else {
            ConsolidationPhase::Transition
        }
    }

    /// Update phase based on elapsed time
    fn update(&mut self) {
        let now = Instant::now();
        let elapsed = (now - self.last_update).as_secs_f32();
        self.last_update = now;

        // Phase advancement with modulation
        let delta_phi = 2.0 * std::f32::consts::PI * self.frequency * self.modulation * elapsed;

        // Oscillate between 0 and PI (half cycle)
        // Using absolute value of sin for [0, PI] range
        let raw_phase = (self.phi + delta_phi) % (2.0 * std::f32::consts::PI);
        self.phi = raw_phase.sin().abs() * std::f32::consts::PI;

        // Ensure valid range
        self.phi = self.phi.clamp(0.0, std::f32::consts::PI);
    }

    /// Apply external modulation (e.g., from neuromodulators)
    pub fn set_modulation(&mut self, modulation: f32) {
        self.modulation = modulation.clamp(0.1, 3.0);
    }

    /// Reset to encoding phase
    pub fn reset_to_encoding(&mut self) {
        self.phi = 0.0;
    }

    /// Jump to consolidation phase (for dream triggers)
    pub fn force_consolidation(&mut self) {
        self.phi = std::f32::consts::PI * 0.9;
    }

    /// Check if consolidation should be triggered
    pub fn should_consolidate(&self, node: &MemoryNode, config: &PhaseConfig) -> bool {
        // High phase angle + sufficient importance + not recently accessed
        let phase_ready = self.phi > config.consolidation_phase_threshold;
        let importance_sufficient = node.importance > config.consolidation_importance_threshold;
        let access_stale = node.last_accessed.elapsed() > config.consolidation_staleness;

        phase_ready && importance_sufficient && access_stale
    }
}
```

### 6.3 Configuration

```yaml
phase:
  # Oscillator parameters
  base_frequency: 0.1       # Cycles per second (slow oscillation)
  modulation_min: 0.1       # Minimum modulation factor
  modulation_max: 3.0       # Maximum modulation factor

  # Phase boundaries (as fraction of PI)
  encoding_threshold: 0.33  # Below this = encoding phase
  consolidation_threshold: 0.67  # Above this = consolidation phase

  # Consolidation triggers
  consolidation_phase_threshold: 2.1  # phi > this for consolidation
  consolidation_importance_threshold: 0.4
  consolidation_staleness_seconds: 300  # 5 minutes
```

---

## 7. Johari Quadrant Classifier

### 7.1 Overview

The Johari Window model classifies memories into four quadrants based on entropy (surprise) and coherence levels, guiding retrieval and processing strategies.

### 7.2 Quadrant Definitions

| Quadrant | delta_s | delta_c | Meaning | System Behavior |
|----------|---------|---------|---------|-----------------|
| **Open** | Low (<0.5) | High (>0.5) | Known to self and others | Direct recall, fast retrieval |
| **Blind** | High (>0.5) | Low (<0.5) | Unknown to self | Discovery zone, trigger exploration |
| **Hidden** | Low (<0.5) | Low (<0.5) | Known to self only | Private knowledge, use get_neighborhood |
| **Unknown** | High (>0.5) | High (>0.5) | Unknown to all | Exploration frontier, epistemic action |

### 7.3 Implementation

```rust
/// Johari quadrant for memory classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JohariQuadrant {
    /// Low entropy, high coherence - direct recall
    Open,
    /// High entropy, low coherence - discovery zone
    Blind,
    /// Low entropy, low coherence - private knowledge
    Hidden,
    /// High entropy, high coherence - exploration frontier
    Unknown,
}

pub struct JohariClassifier {
    /// Threshold for entropy (delta_s) classification
    entropy_threshold: f32,
    /// Threshold for coherence (delta_c) classification
    coherence_threshold: f32,
}

impl JohariClassifier {
    pub fn new(config: &JohariConfig) -> Self {
        Self {
            entropy_threshold: config.entropy_threshold,
            coherence_threshold: config.coherence_threshold,
        }
    }

    /// Classify memory into Johari quadrant
    pub fn classify(&self, delta_s: f32, delta_c: f32) -> JohariQuadrant {
        let low_entropy = delta_s < self.entropy_threshold;
        let high_coherence = delta_c > self.coherence_threshold;

        match (low_entropy, high_coherence) {
            (true, true) => JohariQuadrant::Open,
            (false, false) => JohariQuadrant::Blind,
            (true, false) => JohariQuadrant::Hidden,
            (false, true) => JohariQuadrant::Unknown,
        }
    }

    /// Get suggested action for quadrant
    pub fn suggested_action(&self, quadrant: JohariQuadrant) -> SuggestedAction {
        match quadrant {
            JohariQuadrant::Open => SuggestedAction::DirectRecall,
            JohariQuadrant::Blind => SuggestedAction::TriggerDream,
            JohariQuadrant::Hidden => SuggestedAction::GetNeighborhood,
            JohariQuadrant::Unknown => SuggestedAction::EpistemicAction,
        }
    }

    /// Get retrieval strategy for quadrant
    pub fn retrieval_strategy(&self, quadrant: JohariQuadrant) -> RetrievalStrategy {
        match quadrant {
            JohariQuadrant::Open => RetrievalStrategy {
                search_depth: 1,
                include_neighbors: false,
                confidence_threshold: 0.8,
                max_results: 5,
            },
            JohariQuadrant::Blind => RetrievalStrategy {
                search_depth: 3,
                include_neighbors: true,
                confidence_threshold: 0.5,
                max_results: 20,
            },
            JohariQuadrant::Hidden => RetrievalStrategy {
                search_depth: 2,
                include_neighbors: true,
                confidence_threshold: 0.6,
                max_results: 10,
            },
            JohariQuadrant::Unknown => RetrievalStrategy {
                search_depth: 4,
                include_neighbors: true,
                confidence_threshold: 0.4,
                max_results: 30,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum SuggestedAction {
    DirectRecall,
    TriggerDream,
    GetNeighborhood,
    EpistemicAction,
    CritiqueContext,
    Curate,
}

#[derive(Debug, Clone)]
pub struct RetrievalStrategy {
    pub search_depth: u32,
    pub include_neighbors: bool,
    pub confidence_threshold: f32,
    pub max_results: usize,
}
```

### 7.4 Configuration

```yaml
johari:
  # Thresholds for quadrant classification
  entropy_threshold: 0.5    # Below = low entropy
  coherence_threshold: 0.5  # Above = high coherence

  # Retrieval configurations per quadrant
  strategies:
    open:
      search_depth: 1
      include_neighbors: false
      confidence_threshold: 0.8
      max_results: 5
    blind:
      search_depth: 3
      include_neighbors: true
      confidence_threshold: 0.5
      max_results: 20
    hidden:
      search_depth: 2
      include_neighbors: true
      confidence_threshold: 0.6
      max_results: 10
    unknown:
      search_depth: 4
      include_neighbors: true
      confidence_threshold: 0.4
      max_results: 30
```

---

## 8. System Lifecycle State Machine

### 8.1 Overview

The system transitions through lifecycle phases based on interaction count, adjusting thresholds and behavior accordingly.

### 8.2 Lifecycle Phases (Marblestone Dynamic Learning)

The lifecycle phases implement Marblestone-inspired dynamic learning rates, where lambda weights adjust the balance between novelty seeking (entropy/surprise) and coherence preservation based on system maturity.

| Phase | Interactions | Entropy Trigger | Coherence Trigger | lambda_delta_S | lambda_delta_C | Stance |
|-------|--------------|-----------------|-------------------|---------------|---------------|--------|
| **Infancy** | 0-50 | 0.9 | 0.2 | 0.7 | 0.3 | Capture-Heavy (high novelty seeking) |
| **Growth** | 50-500 | 0.7 | 0.4 | 0.5 | 0.5 | Balanced (exploration and consolidation) |
| **Maturity** | 500+ | 0.6 | 0.5 | 0.3 | 0.7 | Curation-Heavy (coherence preservation) |

**Marblestone Dynamics**:
- **Infancy**: High novelty seeking (lambda_delta_S=0.7), low coherence requirement (lambda_delta_C=0.3) - prioritizes capturing new information
- **Growth**: Balanced exploration and consolidation (lambda_delta_S=0.5, lambda_delta_C=0.5) - optimal learning phase
- **Maturity**: Low novelty seeking (lambda_delta_S=0.3), high coherence preservation (lambda_delta_C=0.7) - prioritizes stability

### 8.2.1 Requirements

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| REQ-UTL-030 | System SHALL adjust lambda_delta_S/lambda_delta_C based on lifecycle stage | Must | Marblestone dynamic learning |
| REQ-UTL-031 | Infancy stage SHALL use lambda_delta_S=0.7, lambda_delta_C=0.3 | Must | High novelty seeking in early learning |
| REQ-UTL-032 | Growth stage SHALL use lambda_delta_S=0.5, lambda_delta_C=0.5 | Must | Balanced exploration and consolidation |
| REQ-UTL-033 | Maturity stage SHALL use lambda_delta_S=0.3, lambda_delta_C=0.7 | Must | Coherence preservation in stable operation |
| REQ-UTL-034 | Lambda weights SHALL be applied to UTL learning magnitude computation | Must | Dynamic modulation of learning signal |
| REQ-UTL-035 | Lifecycle transitions SHALL preserve accumulated knowledge coherence | Should | Prevent learning regression |

### 8.3 Implementation

```rust
/// System lifecycle stage with Marblestone-inspired dynamic learning rates
///
/// Different stages emphasize different aspects of learning:
/// - Infancy: High novelty seeking (lambda_delta_S), low coherence requirement
/// - Growth: Balanced exploration and consolidation
/// - Maturity: Low novelty seeking, high coherence preservation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleStage {
    /// Early learning: prioritize novelty (lambda_delta_S=0.7, lambda_delta_C=0.3)
    Infancy,
    /// Active learning: balanced (lambda_delta_S=0.5, lambda_delta_C=0.5)
    Growth,
    /// Stable operation: prioritize coherence (lambda_delta_S=0.3, lambda_delta_C=0.7)
    Maturity,
}

/// Dynamic lambda weights based on lifecycle stage (Marblestone)
///
/// These weights modulate the balance between surprise/novelty (delta_S)
/// and coherence (delta_C) in the UTL learning equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifecycleLambdaWeights {
    /// Weight for entropy/surprise component (lambda_delta_S)
    pub lambda_delta_s: f32,
    /// Weight for coherence component (lambda_delta_C)
    pub lambda_delta_c: f32,
}

impl LifecycleStage {
    /// Get lambda weights for current lifecycle stage
    ///
    /// Returns Marblestone-inspired dynamic weights that adjust
    /// the learning signal computation based on system maturity.
    pub fn get_lambda_weights(&self) -> LifecycleLambdaWeights {
        match self {
            LifecycleStage::Infancy => LifecycleLambdaWeights {
                lambda_delta_s: 0.7,
                lambda_delta_c: 0.3,
            },
            LifecycleStage::Growth => LifecycleLambdaWeights {
                lambda_delta_s: 0.5,
                lambda_delta_c: 0.5,
            },
            LifecycleStage::Maturity => LifecycleLambdaWeights {
                lambda_delta_s: 0.3,
                lambda_delta_c: 0.7,
            },
        }
    }

    /// Check if this stage prioritizes novelty seeking
    pub fn is_novelty_seeking(&self) -> bool {
        self.get_lambda_weights().lambda_delta_s > 0.5
    }

    /// Check if this stage prioritizes coherence preservation
    pub fn is_coherence_preserving(&self) -> bool {
        self.get_lambda_weights().lambda_delta_c > 0.5
    }
}

/// Backward compatibility alias
pub type LifecyclePhase = LifecycleStage;

/// Lifecycle state machine for adaptive thresholds with Marblestone dynamics
pub struct LifecycleManager {
    /// Current interaction count
    interaction_count: u64,
    /// Current lifecycle stage
    current_stage: LifecycleStage,
    /// Phase-specific configuration
    config: LifecycleConfig,
    /// Current lambda weights (cached for performance)
    lambda_weights: LifecycleLambdaWeights,
}

#[derive(Debug, Clone)]
pub struct PhaseThresholds {
    pub entropy_trigger: f32,
    pub coherence_trigger: f32,
    pub min_importance_store: f32,
    pub consolidation_threshold: f32,
}

impl LifecycleManager {
    pub fn new(config: LifecycleConfig) -> Self {
        let initial_stage = LifecycleStage::Infancy;
        Self {
            interaction_count: 0,
            current_stage: initial_stage,
            config,
            lambda_weights: initial_stage.get_lambda_weights(),
        }
    }

    /// Record an interaction and check for phase transition
    pub fn record_interaction(&mut self) {
        self.interaction_count += 1;
        self.check_phase_transition();
    }

    /// Get current lifecycle stage
    pub fn current_stage(&self) -> LifecycleStage {
        self.current_stage
    }

    /// Backward compatibility: Get current lifecycle phase
    pub fn current_phase(&self) -> LifecyclePhase {
        self.current_stage
    }

    /// Get current lambda weights for Marblestone dynamic learning
    ///
    /// Returns the lambda_delta_S and lambda_delta_C weights that should
    /// be applied to the UTL learning magnitude computation.
    pub fn get_lambda_weights(&self) -> LifecycleLambdaWeights {
        self.lambda_weights
    }

    /// Compute weighted learning components using Marblestone dynamics
    ///
    /// Applies lifecycle-specific lambda weights to surprise and coherence
    /// components before they are combined in the UTL equation.
    ///
    /// # Arguments
    /// * `delta_s` - Raw surprise/entropy component
    /// * `delta_c` - Raw coherence component
    ///
    /// # Returns
    /// Tuple of (weighted_delta_s, weighted_delta_c)
    pub fn apply_lambda_weights(&self, delta_s: f32, delta_c: f32) -> (f32, f32) {
        let weights = self.get_lambda_weights();
        (
            delta_s * weights.lambda_delta_s,
            delta_c * weights.lambda_delta_c,
        )
    }

    /// Get current thresholds based on stage
    pub fn get_thresholds(&self) -> PhaseThresholds {
        match self.current_stage {
            LifecycleStage::Infancy => PhaseThresholds {
                entropy_trigger: self.config.infancy.entropy_trigger,
                coherence_trigger: self.config.infancy.coherence_trigger,
                min_importance_store: self.config.infancy.min_importance_store,
                consolidation_threshold: self.config.infancy.consolidation_threshold,
            },
            LifecycleStage::Growth => PhaseThresholds {
                entropy_trigger: self.config.growth.entropy_trigger,
                coherence_trigger: self.config.growth.coherence_trigger,
                min_importance_store: self.config.growth.min_importance_store,
                consolidation_threshold: self.config.growth.consolidation_threshold,
            },
            LifecycleStage::Maturity => PhaseThresholds {
                entropy_trigger: self.config.maturity.entropy_trigger,
                coherence_trigger: self.config.maturity.coherence_trigger,
                min_importance_store: self.config.maturity.min_importance_store,
                consolidation_threshold: self.config.maturity.consolidation_threshold,
            },
        }
    }

    /// Check if phase transition should occur
    fn check_phase_transition(&mut self) {
        let new_stage = if self.interaction_count < self.config.infancy_threshold {
            LifecycleStage::Infancy
        } else if self.interaction_count < self.config.growth_threshold {
            LifecycleStage::Growth
        } else {
            LifecycleStage::Maturity
        };

        if new_stage != self.current_stage {
            let old_weights = self.lambda_weights;
            let new_weights = new_stage.get_lambda_weights();

            log::info!(
                "Lifecycle transition: {:?} -> {:?} at {} interactions",
                self.current_stage, new_stage, self.interaction_count
            );
            log::info!(
                "Lambda weights change: (lambda_dS={:.2}, lambda_dC={:.2}) -> (lambda_dS={:.2}, lambda_dC={:.2})",
                old_weights.lambda_delta_s, old_weights.lambda_delta_c,
                new_weights.lambda_delta_s, new_weights.lambda_delta_c
            );

            self.current_stage = new_stage;
            self.lambda_weights = new_weights;
        }
    }

    /// Get storage stance based on current stage
    pub fn storage_stance(&self) -> StorageStance {
        match self.current_stage {
            LifecycleStage::Infancy => StorageStance::CaptureHeavy,
            LifecycleStage::Growth => StorageStance::Balanced,
            LifecycleStage::Maturity => StorageStance::CurationHeavy,
        }
    }

    /// Check if content should be stored based on phase
    pub fn should_store(&self, learning_magnitude: f32, importance: f32) -> bool {
        let thresholds = self.get_thresholds();
        match self.storage_stance() {
            StorageStance::CaptureHeavy => {
                // Store almost everything
                learning_magnitude > 0.1 || importance > thresholds.min_importance_store
            }
            StorageStance::Balanced => {
                // Standard threshold
                learning_magnitude > thresholds.consolidation_threshold
                || importance > thresholds.min_importance_store
            }
            StorageStance::CurationHeavy => {
                // High bar for storage
                learning_magnitude > thresholds.consolidation_threshold
                && importance > thresholds.min_importance_store
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageStance {
    CaptureHeavy,
    Balanced,
    CurationHeavy,
}
```

### 8.4 Configuration

```yaml
lifecycle:
  # Phase boundaries
  infancy_threshold: 50     # Interactions before Growth
  growth_threshold: 500     # Interactions before Maturity

  # Marblestone dynamic lambda weights (adjust novelty vs coherence emphasis)
  # Lambda weights must sum to 1.0 for each stage
  lambda_weights:
    infancy:
      lambda_delta_s: 0.7   # High novelty seeking
      lambda_delta_c: 0.3   # Low coherence requirement
    growth:
      lambda_delta_s: 0.5   # Balanced exploration
      lambda_delta_c: 0.5   # Balanced consolidation
    maturity:
      lambda_delta_s: 0.3   # Low novelty seeking
      lambda_delta_c: 0.7   # High coherence preservation

  # Infancy phase (0-50 interactions)
  infancy:
    entropy_trigger: 0.9
    coherence_trigger: 0.2
    min_importance_store: 0.1
    consolidation_threshold: 0.3
    stance: "capture_heavy"

  # Growth phase (50-500 interactions)
  growth:
    entropy_trigger: 0.7
    coherence_trigger: 0.4
    min_importance_store: 0.3
    consolidation_threshold: 0.5
    stance: "balanced"

  # Maturity phase (500+ interactions)
  maturity:
    entropy_trigger: 0.6
    coherence_trigger: 0.5
    min_importance_store: 0.4
    consolidation_threshold: 0.6
    stance: "curation_heavy"
```

---

## 9. UTLProcessor Implementation

### 9.1 Core Structure

```rust
/// Main UTL processor integrating all components
pub struct UTLProcessor {
    /// Surprise computation
    surprise_calculator: SurpriseCalculator,
    /// Coherence tracking
    coherence_tracker: CoherenceTracker,
    /// Emotional weighting
    emotional_calculator: EmotionalWeightCalculator,
    /// Phase oscillator
    phase_oscillator: PhaseOscillator,
    /// Johari classifier
    johari_classifier: JohariClassifier,
    /// Lifecycle manager
    lifecycle_manager: LifecycleManager,
    /// Configuration
    config: UTLConfig,
}

/// Learning signal output from UTL computation
#[derive(Debug, Clone)]
pub struct LearningSignal {
    /// Learning magnitude L in [0, 1]
    pub magnitude: f32,
    /// Surprise component delta_s
    pub delta_s: f32,
    /// Coherence component delta_c
    pub delta_c: f32,
    /// Emotional weight w_e
    pub w_e: f32,
    /// Phase angle phi
    pub phi: f32,
    /// Johari quadrant classification
    pub quadrant: JohariQuadrant,
    /// Suggested action
    pub suggested_action: SuggestedAction,
    /// Should this trigger consolidation?
    pub should_consolidate: bool,
    /// Should this be stored?
    pub should_store: bool,
    /// Computation timestamp
    pub timestamp: DateTime<Utc>,
}

impl UTLProcessor {
    pub fn new(config: UTLConfig, graph: Arc<RwLock<KnowledgeGraph>>) -> Self {
        Self {
            surprise_calculator: SurpriseCalculator::new(&config.surprise),
            coherence_tracker: CoherenceTracker::new(&config.coherence, graph),
            emotional_calculator: EmotionalWeightCalculator::new(&config.emotional),
            phase_oscillator: PhaseOscillator::new(&config.phase),
            johari_classifier: JohariClassifier::new(&config.johari),
            lifecycle_manager: LifecycleManager::new(config.lifecycle.clone()),
            config,
        }
    }

    /// Compute learning signal for input
    ///
    /// # Arguments
    /// * `input` - New memory content
    /// * `embedding` - Pre-computed embedding for input
    /// * `context` - Current session context
    ///
    /// # Returns
    /// Complete learning signal with all UTL components
    ///
    /// # Constraint: Operation_Latency < 10ms
    pub fn compute_learning(
        &mut self,
        input: &str,
        embedding: &[f32; 1536],
        context: &SessionContext,
    ) -> Result<LearningSignal, UTLError> {
        // Record interaction for lifecycle
        self.lifecycle_manager.record_interaction();

        // Get current lifecycle thresholds
        let thresholds = self.lifecycle_manager.get_thresholds();

        // 1. Compute surprise (delta_s)
        let context_embeddings: Vec<&[f32; 1536]> = context
            .recent_nodes
            .iter()
            .map(|n| &n.embedding)
            .collect();
        let delta_s = self.surprise_calculator
            .compute_surprise_ensemble(embedding, &context_embeddings);

        // 2. Compute coherence (delta_c)
        let delta_c = self.coherence_tracker
            .compute_coherence(embedding, input, &self.config.coherence);

        // 3. Compute emotional weight (w_e)
        let w_e = self.emotional_calculator
            .compute_weight(input, &self.config.emotional);

        // 4. Get phase angle (phi)
        let phi = self.phase_oscillator.get_phi();

        // 5. Compute learning magnitude
        let magnitude = compute_learning_magnitude(delta_s, delta_c, w_e, phi);

        // Validate result (anti-pattern protection)
        if magnitude.is_nan() || magnitude.is_infinite() {
            return Err(UTLError::InvalidComputation {
                delta_s,
                delta_c,
                w_e,
                phi,
                reason: "Learning magnitude resulted in NaN or Infinity".to_string(),
            });
        }

        // 6. Classify Johari quadrant
        let quadrant = self.johari_classifier.classify(delta_s, delta_c);
        let suggested_action = self.johari_classifier.suggested_action(quadrant);

        // 7. Determine consolidation and storage
        let should_consolidate = magnitude > thresholds.consolidation_threshold;
        let should_store = self.lifecycle_manager.should_store(magnitude, delta_s);

        Ok(LearningSignal {
            magnitude,
            delta_s,
            delta_c,
            w_e,
            phi,
            quadrant,
            suggested_action,
            should_consolidate,
            should_store,
            timestamp: Utc::now(),
        })
    }

    /// Check if node should be consolidated
    pub fn should_consolidate(&self, node: &MemoryNode) -> bool {
        let thresholds = self.lifecycle_manager.get_thresholds();

        // Check importance threshold
        if node.importance < thresholds.min_importance_store {
            return false;
        }

        // Check UTL state
        if let Some(utl_state) = &node.utl_state {
            let magnitude = compute_learning_magnitude(
                utl_state.delta_s,
                utl_state.delta_c,
                utl_state.w_e,
                utl_state.phi,
            );
            magnitude > thresholds.consolidation_threshold
        } else {
            false
        }
    }

    /// Update node salience based on learning signal
    pub fn update_salience(
        &self,
        node: &mut MemoryNode,
        signal: &LearningSignal,
    ) {
        // Only update if signal magnitude is significant
        if signal.magnitude < self.config.thresholds.salience_update_min {
            return;
        }

        // Exponential moving average update
        let alpha = self.config.salience_update_alpha;
        node.importance = alpha * signal.magnitude + (1.0 - alpha) * node.importance;

        // Clamp to valid range
        node.importance = node.importance.clamp(0.0, 1.0);

        // Update UTL state on node
        node.utl_state = Some(UTLState {
            delta_s: signal.delta_s,
            delta_c: signal.delta_c,
            w_e: signal.w_e,
            phi: signal.phi,
            last_computed: signal.timestamp,
        });

        // Update Johari quadrant
        node.johari_quadrant = signal.quadrant;

        // Update access time
        node.last_accessed = Utc::now();
    }

    /// Get current UTL status for reporting
    pub fn get_status(&self) -> UTLStatus {
        UTLStatus {
            lifecycle_stage: self.lifecycle_manager.current_stage(),
            lifecycle_phase: self.lifecycle_manager.current_phase(),
            interaction_count: self.lifecycle_manager.interaction_count,
            current_thresholds: self.lifecycle_manager.get_thresholds(),
            current_phase: self.phase_oscillator.get_phase(),
            phase_angle: self.phase_oscillator.phi,
            emotional_state: self.emotional_calculator.current_state.clone(),
            lambda_weights: self.lifecycle_manager.get_lambda_weights(),
        }
    }

    /// Update coherence window with new entry
    pub fn update_coherence_window(&mut self, entry: CoherenceEntry) {
        self.coherence_tracker.update(entry);
    }

    /// Set phase modulation (from neuromodulation)
    pub fn set_phase_modulation(&mut self, modulation: f32) {
        self.phase_oscillator.set_modulation(modulation);
    }

    /// Force consolidation phase (for dream triggers)
    pub fn trigger_consolidation(&mut self) {
        self.phase_oscillator.force_consolidation();
    }
}

#[derive(Debug, Clone)]
pub struct UTLState {
    pub delta_s: f32,
    pub delta_c: f32,
    pub w_e: f32,
    pub phi: f32,
    pub last_computed: DateTime<Utc>,
}

#[derive(Debug)]
pub struct UTLStatus {
    pub lifecycle_stage: LifecycleStage,
    /// Backward compatibility alias
    pub lifecycle_phase: LifecyclePhase,
    pub interaction_count: u64,
    pub current_thresholds: PhaseThresholds,
    pub current_phase: ConsolidationPhase,
    pub phase_angle: f32,
    pub emotional_state: EmotionalState,
    /// Marblestone dynamic lambda weights for current lifecycle stage
    pub lambda_weights: LifecycleLambdaWeights,
}

#[derive(Debug, thiserror::Error)]
pub enum UTLError {
    #[error("Invalid UTL computation: delta_s={delta_s}, delta_c={delta_c}, w_e={w_e}, phi={phi}. {reason}")]
    InvalidComputation {
        delta_s: f32,
        delta_c: f32,
        w_e: f32,
        phi: f32,
        reason: String,
    },

    #[error("Missing context for computation")]
    MissingContext,

    #[error("Graph access error: {0}")]
    GraphError(String),
}
```

---

## 10. MCP Tool Integration

### 10.1 utl_status Tool

```rust
/// MCP tool: Get UTL metrics for session
///
/// Returns current UTL state including learning metrics,
/// lifecycle phase, and consolidation state.
#[derive(Debug, Serialize, Deserialize)]
pub struct UtlStatusRequest {
    pub session_id: Option<Uuid>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UtlStatusResponse {
    pub lifecycle_phase: String,
    pub interaction_count: u64,
    pub entropy: f32,
    pub coherence: f32,
    pub learning_score: f32,
    pub johari_quadrant: String,
    pub consolidation_phase: String,
    pub phase_angle: f32,
    pub emotional_state: EmotionalStateResponse,
    pub thresholds: ThresholdsResponse,
    pub suggested_action: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmotionalStateResponse {
    pub valence: f32,
    pub arousal: f32,
    pub weight: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThresholdsResponse {
    pub entropy_trigger: f32,
    pub coherence_trigger: f32,
    pub min_importance: f32,
    pub consolidation: f32,
}
```

### 10.2 get_memetic_status Integration

The `get_memetic_status` tool includes UTL metrics in its response:

```rust
pub struct MemeticStatusResponse {
    // ... existing fields ...

    /// UTL-specific metrics
    pub utl_metrics: UTLMetrics,
}

pub struct UTLMetrics {
    pub entropy: f32,
    pub coherence: f32,
    pub learning_score: f32,
    pub johari_quadrant: String,
    pub phase: String,
}
```

### 10.3 Cognitive Pulse Header

Every MCP response includes UTL-driven Cognitive Pulse:

```rust
pub struct CognitivePulse {
    /// Current entropy level
    pub entropy: f32,
    /// Current coherence level
    pub coherence: f32,
    /// UTL-computed learning score
    pub learning_score: f32,
    /// Current Johari quadrant
    pub quadrant: JohariQuadrant,
    /// Suggested action based on UTL state
    pub suggested_action: SuggestedAction,
}
```

---

## 11. Data Structures

### 11.1 MemoryNode UTL Extension

```rust
pub struct MemoryNode {
    // ... existing fields from Module 2 ...

    /// UTL state for this node
    pub utl_state: Option<UTLState>,

    /// Johari quadrant classification
    pub johari_quadrant: JohariQuadrant,

    /// Observer perspective for multi-agent safety
    pub observer_perspective: Option<ObserverPerspective>,

    /// Priors for merge safety
    pub priors_vibe_check: Option<PriorsVibeCheck>,
}

pub struct ObserverPerspective {
    pub domain: String,
    pub confidence_priors: HashMap<String, f32>,
}

pub struct PriorsVibeCheck {
    pub assumption_embedding: [f32; 128],
    pub domain_priors: Vec<String>,
    pub prior_confidence: f32,
}
```

### 11.2 Session Context

```rust
pub struct SessionContext {
    pub session_id: Uuid,
    pub recent_nodes: Vec<MemoryNode>,
    pub current_entropy: f32,
    pub current_coherence: f32,
    pub interaction_count: u64,
    pub last_activity: DateTime<Utc>,
}
```

---

## 12. Testing Requirements

### 12.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_magnitude_basic() {
        // Basic computation
        let l = compute_learning_magnitude(0.5, 0.5, 1.0, 0.0);
        assert!(l > 0.0 && l < 1.0);

        // cos(0) = 1, should maximize learning
        let l_max_phase = compute_learning_magnitude(1.0, 1.0, 1.5, 0.0);
        assert!(l_max_phase > 0.5);

        // cos(PI) = -1, should minimize learning
        let l_min_phase = compute_learning_magnitude(1.0, 1.0, 1.5, std::f32::consts::PI);
        assert!(l_min_phase < 0.5);
    }

    #[test]
    fn test_learning_magnitude_clamping() {
        // Out of range inputs should be clamped
        let l = compute_learning_magnitude(-0.5, 2.0, 0.1, 10.0);
        assert!(l >= 0.0 && l <= 1.0);
        assert!(!l.is_nan());
        assert!(!l.is_infinite());
    }

    #[test]
    fn test_no_nan_infinity() {
        // Edge cases that might produce NaN/Infinity
        let cases = [
            (0.0, 0.0, 0.5, 0.0),
            (1.0, 1.0, 1.5, std::f32::consts::PI),
            (f32::MIN_POSITIVE, f32::MIN_POSITIVE, 0.5, 0.0),
        ];

        for (ds, dc, we, phi) in cases {
            let l = compute_learning_magnitude(ds, dc, we, phi);
            assert!(!l.is_nan(), "NaN for inputs: {:?}", (ds, dc, we, phi));
            assert!(!l.is_infinite(), "Infinity for inputs: {:?}", (ds, dc, we, phi));
        }
    }

    #[test]
    fn test_johari_classification() {
        let classifier = JohariClassifier::new(&JohariConfig::default());

        assert_eq!(classifier.classify(0.3, 0.7), JohariQuadrant::Open);
        assert_eq!(classifier.classify(0.7, 0.3), JohariQuadrant::Blind);
        assert_eq!(classifier.classify(0.3, 0.3), JohariQuadrant::Hidden);
        assert_eq!(classifier.classify(0.7, 0.7), JohariQuadrant::Unknown);
    }

    #[test]
    fn test_lifecycle_transitions() {
        let config = LifecycleConfig::default();
        let mut manager = LifecycleManager::new(config);

        assert_eq!(manager.current_phase(), LifecyclePhase::Infancy);

        // Simulate 50 interactions
        for _ in 0..50 {
            manager.record_interaction();
        }
        assert_eq!(manager.current_phase(), LifecyclePhase::Growth);

        // Simulate 450 more interactions
        for _ in 0..450 {
            manager.record_interaction();
        }
        assert_eq!(manager.current_phase(), LifecyclePhase::Maturity);
    }

    #[test]
    fn test_lifecycle_stage_lambda_weights() {
        // Test Marblestone dynamic lambda weights for each stage

        // Infancy: High novelty seeking
        let infancy_weights = LifecycleStage::Infancy.get_lambda_weights();
        assert_eq!(infancy_weights.lambda_delta_s, 0.7);
        assert_eq!(infancy_weights.lambda_delta_c, 0.3);
        assert!(LifecycleStage::Infancy.is_novelty_seeking());
        assert!(!LifecycleStage::Infancy.is_coherence_preserving());

        // Growth: Balanced
        let growth_weights = LifecycleStage::Growth.get_lambda_weights();
        assert_eq!(growth_weights.lambda_delta_s, 0.5);
        assert_eq!(growth_weights.lambda_delta_c, 0.5);
        assert!(!LifecycleStage::Growth.is_novelty_seeking());
        assert!(!LifecycleStage::Growth.is_coherence_preserving());

        // Maturity: Coherence preserving
        let maturity_weights = LifecycleStage::Maturity.get_lambda_weights();
        assert_eq!(maturity_weights.lambda_delta_s, 0.3);
        assert_eq!(maturity_weights.lambda_delta_c, 0.7);
        assert!(!LifecycleStage::Maturity.is_novelty_seeking());
        assert!(LifecycleStage::Maturity.is_coherence_preserving());
    }

    #[test]
    fn test_lifecycle_manager_lambda_weight_transitions() {
        let config = LifecycleConfig::default();
        let mut manager = LifecycleManager::new(config);

        // Infancy lambda weights
        let weights = manager.get_lambda_weights();
        assert_eq!(weights.lambda_delta_s, 0.7);
        assert_eq!(weights.lambda_delta_c, 0.3);

        // Transition to Growth
        for _ in 0..50 {
            manager.record_interaction();
        }
        let weights = manager.get_lambda_weights();
        assert_eq!(weights.lambda_delta_s, 0.5);
        assert_eq!(weights.lambda_delta_c, 0.5);

        // Transition to Maturity
        for _ in 0..450 {
            manager.record_interaction();
        }
        let weights = manager.get_lambda_weights();
        assert_eq!(weights.lambda_delta_s, 0.3);
        assert_eq!(weights.lambda_delta_c, 0.7);
    }

    #[test]
    fn test_apply_lambda_weights() {
        let config = LifecycleConfig::default();
        let mut manager = LifecycleManager::new(config);

        // Test weight application in Infancy (novelty seeking)
        let (weighted_s, weighted_c) = manager.apply_lambda_weights(1.0, 1.0);
        assert_eq!(weighted_s, 0.7); // 1.0 * 0.7
        assert_eq!(weighted_c, 0.3); // 1.0 * 0.3

        // Transition to Maturity
        for _ in 0..500 {
            manager.record_interaction();
        }

        // Test weight application in Maturity (coherence preserving)
        let (weighted_s, weighted_c) = manager.apply_lambda_weights(1.0, 1.0);
        assert_eq!(weighted_s, 0.3); // 1.0 * 0.3
        assert_eq!(weighted_c, 0.7); // 1.0 * 0.7
    }

    #[test]
    fn test_lambda_weights_sum_to_one() {
        // Verify lambda weights always sum to 1.0 (conservation property)
        for stage in [LifecycleStage::Infancy, LifecycleStage::Growth, LifecycleStage::Maturity] {
            let weights = stage.get_lambda_weights();
            let sum = weights.lambda_delta_s + weights.lambda_delta_c;
            assert!(
                (sum - 1.0).abs() < f32::EPSILON,
                "Lambda weights should sum to 1.0 for {:?}, got {}",
                stage,
                sum
            );
        }
    }

    #[test]
    fn test_emotional_weight_bounds() {
        let mut calculator = EmotionalWeightCalculator::new(&EmotionalConfig::default());

        // Test various content types
        let test_cases = [
            "Normal text without emotion",
            "URGENT!!! CRITICAL ERROR!!!",
            "Everything is wonderful and excellent!",
            "Failed. Error. Danger ahead.",
            "",
        ];

        for content in test_cases {
            let weight = calculator.compute_weight(content, &EmotionalConfig::default());
            assert!(weight >= 0.5, "Weight too low for: {}", content);
            assert!(weight <= 1.5, "Weight too high for: {}", content);
        }
    }
}
```

### 12.2 Integration Tests

```rust
#[tokio::test]
async fn test_utl_processor_full_pipeline() {
    let graph = create_test_graph().await;
    let config = UTLConfig::default();
    let mut processor = UTLProcessor::new(config, graph);

    // Simulate processing multiple inputs
    let inputs = vec![
        ("Hello, this is a test", generate_random_embedding()),
        ("This relates to the test above", generate_random_embedding()),
        ("CRITICAL: System failure!", generate_random_embedding()),
    ];

    let context = SessionContext::new_empty();

    for (input, embedding) in inputs {
        let signal = processor.compute_learning(input, &embedding, &context).unwrap();

        // Verify signal validity
        assert!(signal.magnitude >= 0.0 && signal.magnitude <= 1.0);
        assert!(signal.delta_s >= 0.0 && signal.delta_s <= 1.0);
        assert!(signal.delta_c >= 0.0 && signal.delta_c <= 1.0);
        assert!(signal.w_e >= 0.5 && signal.w_e <= 1.5);
        assert!(signal.phi >= 0.0 && signal.phi <= std::f32::consts::PI);
    }
}

#[tokio::test]
async fn test_learning_signal_importance_correlation() {
    // Quality gate: Learning signal should correlate with importance (r > 0.7)
    let graph = create_test_graph().await;
    let config = UTLConfig::default();
    let mut processor = UTLProcessor::new(config, graph);

    // Generate test data with known importance levels
    let test_data = generate_importance_test_data(100);
    let context = SessionContext::new_empty();

    let mut importances = Vec::new();
    let mut learning_scores = Vec::new();

    for (content, embedding, expected_importance) in test_data {
        let signal = processor.compute_learning(&content, &embedding, &context).unwrap();
        importances.push(expected_importance);
        learning_scores.push(signal.magnitude);
    }

    let correlation = pearson_correlation(&importances, &learning_scores);
    assert!(correlation > 0.7, "Learning signal correlation with importance: {}", correlation);
}
```

### 12.3 Benchmark Tests

```rust
#[bench]
fn bench_compute_learning_magnitude(b: &mut Bencher) {
    b.iter(|| {
        compute_learning_magnitude(0.5, 0.5, 1.0, 1.0)
    });
}

#[bench]
fn bench_full_utl_computation(b: &mut Bencher) {
    let graph = create_test_graph_sync();
    let config = UTLConfig::default();
    let mut processor = UTLProcessor::new(config, graph);
    let context = SessionContext::new_empty();
    let embedding = generate_random_embedding();

    b.iter(|| {
        processor.compute_learning("Test input", &embedding, &context).unwrap()
    });
}
```

---

## 13. Quality Gates

### 13.1 Functional Requirements

| Requirement | Metric | Target |
|-------------|--------|--------|
| Learning signal validity | All outputs in [0,1] | 100% |
| No NaN/Infinity | Invalid computation count | 0 |
| Importance correlation | Pearson r with human ratings | > 0.7 |
| Surprise detection | Novel content identified | > 90% |
| Coherence stability | Variance over time | < 0.1 |
| Emotional detection | Sentiment accuracy | > 80% |

### 13.2 Performance Requirements

| Operation | Latency Target | P99 Target |
|-----------|---------------|------------|
| compute_learning_magnitude | < 100us | < 500us |
| Full UTL computation | < 10ms | < 50ms |
| Surprise calculation | < 5ms | < 20ms |
| Coherence calculation | < 5ms | < 25ms |
| Emotional weight | < 1ms | < 5ms |
| Phase oscillator update | < 10us | < 50us |

### 13.3 Test Coverage

- Unit test coverage: > 90%
- Integration test coverage: > 80%
- All edge cases documented and tested
- All anti-patterns have prevention tests

---

## 14. Anti-Patterns and Mitigations

### 14.1 AP-009: NaN/Infinity in UTL Calculations

**Problem**: Floating point operations can produce NaN or Infinity

**Mitigation**:
```rust
// Always clamp inputs to valid ranges
let delta_s = delta_s.clamp(0.0, 1.0);
let delta_c = delta_c.clamp(0.0, 1.0);
let w_e = w_e.clamp(0.5, 1.5);
let phi = phi.clamp(0.0, std::f32::consts::PI);

// Check output validity
if result.is_nan() || result.is_infinite() {
    return Err(UTLError::InvalidComputation { ... });
}
```

### 14.2 Hardcoded Values

**Problem**: Magic numbers make tuning difficult

**Mitigation**: All thresholds in configuration files:
```yaml
# All values configurable, no hardcoding
utl:
  thresholds:
    consolidation_trigger: 0.7  # Configurable
```

### 14.3 Missing Context Handling

**Problem**: UTL computation with empty context can fail

**Mitigation**:
```rust
if context_embeddings.is_empty() {
    return config.max_surprise_no_context;  // Default value
}
```

---

## 15. Configuration Reference

### 15.1 Complete Configuration Schema

```yaml
# config/utl.yaml - Complete UTL configuration

utl:
  # Core equation
  learning_scale_factor: 2.0

  # Loss function weights
  weights:
    lambda_task: 0.4
    lambda_semantic: 0.3
    lambda_dyn: 0.3

  # Valid ranges for inputs
  ranges:
    delta_s: { min: 0.0, max: 1.0 }
    delta_c: { min: 0.0, max: 1.0 }
    w_e: { min: 0.5, max: 1.5 }
    phi: { min: 0.0, max: 3.14159265 }

  # Thresholds
  thresholds:
    consolidation_trigger: 0.7
    salience_update_min: 0.1
    surprise_significant: 0.6

  # Salience update
  salience_update_alpha: 0.3

surprise:
  kl_weight: 0.6
  distance_weight: 0.4
  kl:
    epsilon: 1e-10
    max_kl_value: 10.0
    temperature: 1.0
  context_window_size: 50
  context_decay: 0.95
  max_surprise_no_context: 0.9
  min_context_for_kl: 3

coherence:
  window_size: 100
  recency_decay: 0.98
  semantic_weight: 0.6
  structural_weight: 0.4
  contradiction_search_k: 20
  contradiction_similarity_threshold: 0.85
  max_contradiction_penalty: 0.5
  default_coherence_empty: 0.5
  default_coherence_no_concepts: 0.4
  max_edge_normalization: 50.0

emotional:
  decay_rate: 0.1
  baseline_weight: 1.0
  valence_weight: 0.6
  arousal_weight: 0.4
  intensity_scale: 0.5
  exclamation_weight: 0.3
  question_weight: 0.2
  caps_weight: 0.2
  weight_min: 0.5
  weight_max: 1.5

phase:
  base_frequency: 0.1
  modulation_min: 0.1
  modulation_max: 3.0
  encoding_threshold: 0.33
  consolidation_threshold: 0.67
  consolidation_phase_threshold: 2.1
  consolidation_importance_threshold: 0.4
  consolidation_staleness_seconds: 300

johari:
  entropy_threshold: 0.5
  coherence_threshold: 0.5

lifecycle:
  infancy_threshold: 50
  growth_threshold: 500

  # Marblestone dynamic lambda weights
  lambda_weights:
    infancy:
      lambda_delta_s: 0.7   # High novelty seeking
      lambda_delta_c: 0.3   # Low coherence requirement
    growth:
      lambda_delta_s: 0.5   # Balanced
      lambda_delta_c: 0.5   # Balanced
    maturity:
      lambda_delta_s: 0.3   # Low novelty seeking
      lambda_delta_c: 0.7   # High coherence preservation

  infancy:
    entropy_trigger: 0.9
    coherence_trigger: 0.2
    min_importance_store: 0.1
    consolidation_threshold: 0.3
  growth:
    entropy_trigger: 0.7
    coherence_trigger: 0.4
    min_importance_store: 0.3
    consolidation_threshold: 0.5
  maturity:
    entropy_trigger: 0.6
    coherence_trigger: 0.5
    min_importance_store: 0.4
    consolidation_threshold: 0.6
```

---

## 16. File Structure

```
crates/context-graph-core/src/
├── utl/
│   ├── mod.rs                 # Module exports
│   ├── processor.rs           # UTLProcessor implementation
│   ├── surprise.rs            # Surprise computation (delta_s)
│   ├── coherence.rs           # Coherence tracking (delta_c)
│   ├── emotional.rs           # Emotional weighting (w_e)
│   ├── phase.rs               # Phase oscillator (phi)
│   ├── johari.rs              # Johari quadrant classifier
│   ├── lifecycle.rs           # System lifecycle state machine
│   ├── types.rs               # Data structures (LearningSignal, UTLState, etc.)
│   ├── config.rs              # Configuration structures
│   └── tests.rs               # Unit tests

config/
├── utl.yaml                   # UTL configuration

tests/integration/
├── utl_integration_test.rs    # Integration tests
├── utl_correlation_test.rs    # Importance correlation tests

benches/
├── utl_bench.rs               # Performance benchmarks
```

---

## 17. Dependencies on Other Modules

### 17.1 Module 4 (Knowledge Graph) - Required

- `KnowledgeGraph` for structural coherence computation
- `search_similar()` for contradiction detection
- `find_contradiction_edge()` for conflict analysis
- Edge traversal for connectivity metrics

### 17.2 Module 3 (Embedding Pipeline) - Required

- Pre-computed embeddings as input
- 1536-dimensional vectors

### 17.3 Module 2 (Core Infrastructure) - Required

- `MemoryNode` data structure
- RocksDB storage for persistence
- Session management

---

## 18. Guidance for Next Agent (Module 6: Bio-Nervous System)

### 18.1 Key Integration Points

1. **UTLProcessor as Learning Layer Component**
   - UTLProcessor should be wrapped as the Learning Layer (L4) in the 5-layer architecture
   - Receives input from Memory Layer (L3)
   - Outputs learning signals to Coherence Layer (L5)

2. **Phase Oscillator Coordination**
   - Phase oscillator state should be shared with other layers
   - Neuromodulation (Module 10) will modulate the oscillator
   - Dream Layer (Module 9) will use `trigger_consolidation()`

3. **Coherence Layer Integration**
   - Coherence metrics from UTL feed into L5 Coherence Layer
   - L5 should use `coherence_tracker` state for global consistency

4. **Latency Budget**
   - UTL computation must complete within Learning Layer budget (<2000ms)
   - Current target: <10ms, leaving headroom for L4 orchestration

### 18.2 Required APIs for Bio-Nervous System

```rust
// UTLProcessor APIs needed by Bio-Nervous System
pub trait UTLInterface {
    fn compute_learning(&mut self, input: &str, embedding: &[f32; 1536], context: &SessionContext) -> Result<LearningSignal, UTLError>;
    fn should_consolidate(&self, node: &MemoryNode) -> bool;
    fn update_salience(&self, node: &mut MemoryNode, signal: &LearningSignal);
    fn get_status(&self) -> UTLStatus;
    fn trigger_consolidation(&mut self);
    fn set_phase_modulation(&mut self, modulation: f32);
}
```

### 18.3 Layer Communication Protocol

The Bio-Nervous System should implement message passing between layers:

```rust
pub struct LayerMessage {
    pub source: NervousLayer,
    pub target: NervousLayer,
    pub payload: MessagePayload,
    pub utl_context: Option<LearningSignal>,  // UTL enrichment
    pub priority: Priority,
    pub deadline: Instant,
}

pub enum MessagePayload {
    // From L3 (Memory) to L4 (Learning)
    MemoryInput { node: MemoryNode, context: SessionContext },

    // From L4 (Learning) to L3 (Memory)
    SalienceUpdate { node_id: Uuid, new_salience: f32 },

    // From L4 (Learning) to L5 (Coherence)
    CoherenceSignal { delta_c: f32, contradictions: Vec<Uuid> },

    // From L5 (Coherence) to L4 (Learning)
    PhaseModulation { modulation: f32 },
}
```

---

## TASK COMPLETION SUMMARY

**What I Did**: Created complete functional specification for Module 5: UTL Integration covering:
- Core UTL equation with mathematical formulations
- Surprise computation via KL divergence and embedding distance
- Coherence tracking with rolling window and graph structure
- Emotional weighting with valence/arousal/decay
- Phase oscillator for consolidation state management
- Johari quadrant classification system
- System lifecycle state machine (Infancy/Growth/Maturity)
- Full UTLProcessor implementation
- MCP tool integration
- Testing requirements with quality gates
- Anti-pattern mitigations (especially NaN/Infinity prevention)
- Complete configuration reference

**Files Created/Modified**:
- `/home/cabdru/contextgraph/specs/functional/module-05-utl-integration.md`

**Memory Locations**:
- specs/functional/module-05-utl-integration.md (specification document)

**Next Agent Guidance**: Module 6 (Bio-Nervous System) needs to:
1. Wrap UTLProcessor as the Learning Layer (L4)
2. Implement layer message passing with UTL enrichment
3. Coordinate phase oscillator with Coherence Layer (L5)
4. Respect latency budgets (<2000ms for Learning Layer)
5. Use provided `UTLInterface` trait for integration
6. Connect LearningSignal outputs to downstream consolidation
