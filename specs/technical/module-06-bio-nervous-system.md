# Module 6: Bio-Nervous System - Technical Specification

```yaml
metadata:
  id: TECH-BIONV-006
  version: 1.0.0
  module: Bio-Nervous System
  phase: 5
  status: draft
  created: 2025-12-31
  dependencies:
    - TECH-UTL-005 (Module 5: UTL Integration)
    - TECH-GRAPH-004 (Module 4: Knowledge Graph)
  functional_spec_ref: SPEC-BIONV-006
  author: Architecture Agent
```

---

## 1. Architecture Overview

### 1.1 5-Layer Architecture

| Layer | Latency Budget | Purpose |
|-------|----------------|---------|
| L1 Sensing | <5ms | Input processing, PII scrubbing, embedding |
| L2 Reflex | <100us | Fast pattern-matched responses |
| L3 Memory | <1ms | Associative storage/retrieval |
| L4 Learning | <10ms | UTL-driven weight optimization |
| L5 Coherence | <10ms | Global state synchronization |

### 1.2 Module Structure

```
crates/context-graph-nervous/src/
├── lib.rs                    # Public API
├── system.rs                 # BioNervousSystem orchestrator
├── message/
│   ├── layer_message.rs      # LayerMessage struct
│   └── payload.rs            # MessagePayload enum
├── layer/
│   └── trait.rs              # NervousLayer trait
├── sensing/                  # L1
├── reflex/                   # L2
├── memory/                   # L3
├── learning/                 # L4
├── coherence/                # L5
└── verification/             # Marblestone formal verification
```

---

## 2. Layer Message Protocol

### 2.1 LayerMessage Struct

```rust
/// Message passed between nervous system layers
#[derive(Debug, Clone)]
pub struct LayerMessage {
    pub id: Uuid,
    pub source: NervousLayerId,
    pub target: NervousLayerId,
    pub payload: MessagePayload,
    pub priority: MessagePriority,
    pub deadline: Instant,
    pub created_at: Instant,
    pub correlation_id: Option<Uuid>,
    pub trace_context: TraceContext,
}

impl LayerMessage {
    pub fn new(
        source: NervousLayerId,
        target: NervousLayerId,
        payload: MessagePayload,
        priority: MessagePriority,
        deadline_budget: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4(),
            source, target, payload, priority,
            deadline: now + deadline_budget,
            created_at: now,
            correlation_id: None,
            trace_context: TraceContext::new(),
        }
    }

    #[inline]
    pub fn is_expired(&self) -> bool { Instant::now() > self.deadline }

    pub fn respond(&self, payload: MessagePayload) -> Self {
        Self {
            id: Uuid::new_v4(),
            source: self.target,
            target: self.source,
            payload,
            priority: self.priority,
            deadline: self.deadline,
            created_at: Instant::now(),
            correlation_id: Some(self.id),
            trace_context: self.trace_context.child(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NervousLayerId {
    Sensing = 1, Reflex = 2, Memory = 3, Learning = 4, Coherence = 5,
}

impl NervousLayerId {
    pub const fn latency_budget(&self) -> Duration {
        match self {
            Self::Sensing => Duration::from_millis(5),
            Self::Reflex => Duration::from_micros(100),
            Self::Memory => Duration::from_millis(1),
            Self::Learning => Duration::from_millis(10),
            Self::Coherence => Duration::from_millis(10),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum MessagePriority {
    Low = 0, Normal = 1, High = 2, Critical = 3, Emergency = 4,
}
```

### 2.2 Message Payloads

```rust
pub enum MessagePayload {
    RawInput(RawInputPayload),
    Embedding(EmbeddingPayload),
    CacheQuery(CacheQueryPayload),
    MemoryQuery(MemoryQueryPayload),
    LearningSignal(LearningSignalPayload),
    CoherenceUpdate(CoherenceUpdatePayload),
    Prediction(PredictionPayload),
    PredictionError(PredictionErrorPayload),
}

pub struct EmbeddingPayload {
    pub embedding: Vector1536,
    pub content: String,
    pub delta_s: f32,
    pub latency_us: u64,
}

pub struct LearningSignalPayload {
    pub delta_s: f32,
    pub delta_c: f32,
    pub w_e: f32,
    pub phi: f32,
    pub learning_score: f32,
}
```

---

## 3. NervousLayer Trait

```rust
#[async_trait]
pub trait NervousLayer: Send + Sync {
    fn id(&self) -> NervousLayerId;
    fn name(&self) -> &'static str { self.id().name() }
    fn latency_budget(&self) -> Duration { self.id().latency_budget() }
    fn is_healthy(&self) -> bool;
    fn health_report(&self) -> HealthReport;
    async fn handle_timeout(&self, message: &LayerMessage) -> TimeoutAction;
    async fn reset(&mut self);
    fn metrics(&self) -> LayerMetrics;
    async fn process_message(&self, message: LayerMessage) -> Result<LayerMessage, LayerError>;
}

pub struct HealthReport {
    pub layer: NervousLayerId,
    pub status: HealthStatus,
    pub latency_p95: Duration,
    pub budget_compliance: f32,
}

pub enum HealthStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
}

pub enum TimeoutAction {
    Partial(Box<dyn Any + Send>),
    Retry { reduced_scope: bool },
    Skip,
    Fail(String),
}
```

---

## 4. Layer Implementations

### 4.1 L1 Sensing (<5ms)

```rust
/// REQ-BIONV-001 through REQ-BIONV-008
pub struct SensingLayer {
    pii_scrubber: PIIScrubber,
    tokenizer: Tokenizer,
    adversarial: AdversarialDetector,
    embedding_pipeline: Arc<dyn EmbeddingProvider>,
    baseline: RwLock<NoveltyBaseline>,
}

impl SensingLayer {
    /// Pipeline: PII Scrub -> Adversarial Check -> Tokenize -> Embed -> Delta-S
    pub async fn process(&self, input: RawInput, ctx: &ProcessingContext) -> Result<SensingOutput, SensingError> {
        let start = Instant::now();
        let scrubbed = self.pii_scrubber.scrub(&input.content);

        let adversarial_check = self.adversarial.check(&scrubbed.content);
        if adversarial_check.is_blocked {
            return Err(SensingError::AdversarialDetected);
        }

        let embedding = self.embedding_pipeline.embed(&scrubbed.content).await?;
        let delta_s = self.compute_delta_s(&embedding).await;

        Ok(SensingOutput { content: scrubbed.content, embedding, delta_s, latency: start.elapsed() })
    }

    async fn compute_delta_s(&self, embedding: &Vector1536) -> f32 {
        let baseline = self.baseline.read().await;
        let distance = 1.0 - cosine_similarity(embedding, &baseline.centroid);
        sigmoid((distance - baseline.mean) / baseline.std.max(0.001))
    }
}
```

### 4.2 L2 Reflex (<100us)

```rust
/// REQ-BIONV-009 through REQ-BIONV-014
pub struct ReflexLayer {
    cache: DashMap<u64, CacheEntry>,
    hopfield: ModernHopfieldNetwork,
    stats: CacheStats,
    config: ReflexConfig,
}

impl ReflexLayer {
    /// Returns Some if cache hit with confidence > 0.95
    pub async fn query(&self, query: &Vector1536, _ctx: &ProcessingContext) -> Option<ReflexResponse> {
        self.stats.lookups.fetch_add(1, Ordering::Relaxed);
        let hash = locality_sensitive_hash(query);

        if let Some(entry) = self.cache.get(&hash) {
            if cosine_similarity(query, &entry.pattern) > self.config.similarity_threshold {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Some(ReflexResponse { response: entry.response.clone(), confidence: entry.confidence });
            }
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    #[inline]
    pub fn should_bypass(&self, confidence: f32) -> bool { confidence > 0.95 }
}
```

### 4.3 L3 Memory (<1ms)

```rust
/// REQ-BIONV-015 through REQ-BIONV-020
pub struct ModernHopfieldNetwork {
    patterns: Vec<Vector1536>,
    metadata: Vec<PatternMetadata>,
    beta: f32,  // [1.0, 5.0], dopamine-controlled
    faiss_index: Option<FAISSGPUIndex>,
}

impl ModernHopfieldNetwork {
    /// Energy: E(x) = -beta * log(sum_i exp(beta * x^T * xi))
    pub fn energy(&self, query: &Vector1536) -> f32 {
        let sum_exp: f64 = self.patterns.iter()
            .map(|p| (self.beta as f64 * dot_product(query, p) as f64).exp())
            .sum();
        -(self.beta as f64 * sum_exp.ln()) as f32
    }

    /// Retrieve: p_i = softmax(beta * query^T * patterns)
    pub fn retrieve(&self, query: &Vector1536, top_k: usize) -> Vec<RetrievedPattern> {
        if let Some(ref index) = self.faiss_index {
            return index.search(query, top_k, &self.patterns, &self.metadata);
        }
        // CPU fallback with softmax attention
        let mut scores: Vec<_> = self.patterns.iter().enumerate()
            .map(|(i, p)| (i, cosine_similarity(query, p)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.into_iter().take(top_k)
            .map(|(i, sim)| RetrievedPattern { pattern: self.patterns[i].clone(), similarity: sim })
            .collect()
    }

    pub fn set_beta(&mut self, beta: f32) { self.beta = beta.clamp(1.0, 5.0); }
}
```

### 4.4 L4 Learning (<10ms)

```rust
/// REQ-BIONV-021 through REQ-BIONV-027
pub struct UTLOptimizer {
    lambda_task: f32,      // 0.4
    lambda_semantic: f32,  // 0.3
    lambda_dyn: f32,       // 0.3
    gradient_clip: f32,    // 1.0
    neuromod: NeuromodulationController,
}

impl UTLOptimizer {
    /// L = f((delta_s * delta_c) * w_e * cos(phi))
    #[inline]
    pub fn compute_learning_score(&self, state: &UTLState) -> f32 {
        let raw = (state.delta_s.clamp(0.0, 1.0) * state.delta_c.clamp(0.0, 1.0))
            * state.w_e.clamp(0.5, 1.5)
            * state.phi.clamp(0.0, PI).cos();
        sigmoid(raw * 2.0)
    }

    /// J = lambda_task * L_task + lambda_semantic * L_semantic + lambda_dyn * (1 - L)
    pub fn compute_loss(&self, state: &UTLState, task_loss: f32, semantic_loss: f32) -> f32 {
        self.lambda_task * task_loss
            + self.lambda_semantic * semantic_loss
            + self.lambda_dyn * (1.0 - self.compute_learning_score(state))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UTLState {
    pub delta_s: f32,  // [0, 1]
    pub delta_c: f32,  // [0, 1]
    pub w_e: f32,      // [0.5, 1.5]
    pub phi: f32,      // [0, PI]
}

/// REQ-BIONV-025, REQ-BIONV-026
pub struct NeuromodulationController {
    pub dopamine: f32,      // -> hopfield.beta [1.0, 5.0]
    pub serotonin: f32,     // -> fuse_moe.top_k [2, 8]
    pub noradrenaline: f32, // -> attention.temperature [0.5, 2.0]
    pub acetylcholine: f32, // -> learning_rate [0.001, 0.002]
}

impl NeuromodulationController {
    pub fn update(&mut self, state: &UTLState) {
        self.dopamine = lerp(self.dopamine, state.delta_s * state.delta_c, 0.1);
        self.serotonin = lerp(self.serotonin, 1.0 - state.delta_s, 0.1);
        self.noradrenaline = lerp(self.noradrenaline, state.delta_s, 0.1);
        self.acetylcholine = lerp(self.acetylcholine, state.delta_c, 0.1);
    }

    pub fn get_hopfield_beta(&self) -> f32 { 1.0 + self.dopamine * 4.0 }
    pub fn get_fuse_moe_top_k(&self) -> usize { 2 + (self.serotonin * 6.0) as usize }
    pub fn get_attention_temperature(&self) -> f32 { 0.5 + self.noradrenaline * 1.5 }
    pub fn get_learning_rate(&self) -> f32 { 0.001 + self.acetylcholine * 0.001 }
}
```

### 4.5 L5 Coherence (<10ms)

```rust
/// REQ-BIONV-028 through REQ-BIONV-035
pub struct CoherenceLayer {
    thalamic_gate: ThalamicGate,
    predictive_coder: PredictiveCoder,
    distiller: ContextDistiller,
    phase: RwLock<f32>,
}

impl CoherenceLayer {
    pub async fn process(&self, input: CoherenceInput, _ctx: &ProcessingContext) -> Result<CoherenceOutput, CoherenceError> {
        if self.thalamic_gate.should_sync() { self.synchronize().await?; }
        let prediction = self.predictive_coder.predict(&input.context_state);
        let coherence_score = self.compute_coherence(&input).await;
        self.update_phase(coherence_score).await;
        let phi = *self.phase.read().await;
        Ok(CoherenceOutput { phi, coherence_score, prediction: Some(prediction) })
    }

    pub async fn get_phase(&self) -> f32 { *self.phase.read().await }

    async fn update_phase(&self, coherence: f32) {
        let mut phase = self.phase.write().await;
        *phase = lerp(*phase, (1.0 - coherence) * FRAC_PI_2, 0.1);
    }
}
```

---

## 5. Formal Verification Layer (Marblestone)

### 5.1 FormalVerificationLayer (REQ-BIONV-056, REQ-BIONV-057)

```rust
/// Lean-inspired formal verification for code nodes
pub struct FormalVerificationLayer {
    smt_solver: SmtSolver,
    verification_cache: LruCache<ContentHash, VerificationStatus>,
    timeout_ms: u64,  // 100ms max
}

impl FormalVerificationLayer {
    pub async fn verify(&mut self, content: &str) -> VerificationStatus {
        if !self.is_code_content(content) { return VerificationStatus::NotApplicable; }

        let hash = ContentHash::compute(content);
        if let Some(status) = self.verification_cache.get(&hash) { return status.clone(); }

        let conditions = self.extract_conditions(content);
        for condition in conditions {
            match self.smt_solver.check(&condition, self.timeout_ms).await {
                SolverResult::Sat(model) => {
                    let status = VerificationStatus::Failed { counterexample: model.to_string() };
                    self.verification_cache.put(hash, status.clone());
                    return status;
                }
                SolverResult::Timeout => {
                    let status = VerificationStatus::Timeout;
                    self.verification_cache.put(hash, status.clone());
                    return status;
                }
                SolverResult::Unsat => continue,
                SolverResult::Unknown => return VerificationStatus::Unknown,
            }
        }
        let status = VerificationStatus::Verified;
        self.verification_cache.put(hash, status.clone());
        status
    }

    fn is_code_content(&self, content: &str) -> bool {
        content.contains("fn ") || content.contains("def ") ||
        content.contains("function ") || content.contains("class ")
    }

    fn extract_conditions(&self, content: &str) -> Vec<VerificationCondition> {
        let mut conditions = Vec::new();
        conditions.extend(self.extract_bounds_checks(content));
        conditions.extend(self.extract_null_checks(content));
        conditions.extend(self.extract_custom_assertions(content));
        conditions
    }
}

pub struct VerificationCondition {
    pub id: Uuid,
    pub assertion: String,               // SMT-LIB2 format
    pub source_location: SourceSpan,
    pub condition_type: ConditionType,
    pub preconditions: Vec<Predicate>,
    pub postconditions: Vec<Predicate>,
    pub invariants: Vec<Predicate>,
}

pub struct Predicate {
    pub expression: String,
    pub description: String,
}

pub enum ConditionType { BoundsCheck, NullSafety, TypeInvariant, Termination, CustomAssertion }

pub enum VerificationStatus {
    Verified,
    Failed { counterexample: String },
    Timeout,
    Unknown,
    NotApplicable,
}
```

### 5.2 SMT Solver Integration

```rust
pub struct SmtSolver {
    backend: SolverBackend,
    config: SolverConfig,
}

impl SmtSolver {
    pub async fn check(&self, condition: &VerificationCondition, timeout_ms: u64) -> SolverResult {
        let smt_input = self.build_smt_input(condition);
        match &self.backend {
            SolverBackend::Z3 => self.run_z3(&smt_input, timeout_ms).await,
            SolverBackend::CVC5 => self.run_cvc5(&smt_input, timeout_ms).await,
            SolverBackend::Portfolio => self.run_portfolio(&smt_input, timeout_ms).await,
        }
    }

    fn build_smt_input(&self, condition: &VerificationCondition) -> String {
        let mut smt = String::from("(set-logic QF_LIA)\n");
        for pre in &condition.preconditions { smt.push_str(&format!("(assert {})\n", pre.expression)); }
        for inv in &condition.invariants { smt.push_str(&format!("(assert {})\n", inv.expression)); }
        smt.push_str(&format!("(assert (not {}))\n", condition.assertion));
        smt.push_str("(check-sat)\n(get-model)\n");
        smt
    }
}

pub enum SolverBackend { Z3, CVC5, Portfolio }
pub enum SolverResult { Sat(Model), Unsat, Timeout, Unknown }

pub struct Model { pub assignments: HashMap<String, Value> }
pub enum Value { Bool(bool), Int(i64), Real(f64), BitVec(Vec<bool>), String(String) }

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    pub fn compute(content: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hasher.finalize());
        ContentHash(hash)
    }
}

pub struct SourceSpan { pub start: usize, pub end: usize, pub line: usize, pub column: usize }
```

---

## 6. Channel Architecture

```rust
pub struct BioNervousSystem {
    sensing: Arc<SensingLayer>,
    reflex: Arc<ReflexLayer>,
    memory: Arc<MemoryLayer>,
    learning: Arc<LearningLayer>,
    coherence: Arc<CoherenceLayer>,
    verification: Arc<RwLock<FormalVerificationLayer>>,
    channels: LayerChannels,
}

pub struct LayerChannels {
    sensing_to_reflex: (mpsc::Sender<LayerMessage>, mpsc::Receiver<LayerMessage>),
    reflex_to_memory: (mpsc::Sender<LayerMessage>, mpsc::Receiver<LayerMessage>),
    memory_to_learning: (mpsc::Sender<LayerMessage>, mpsc::Receiver<LayerMessage>),
    learning_to_coherence: (mpsc::Sender<LayerMessage>, mpsc::Receiver<LayerMessage>),
    coherence_to_sensing: (mpsc::Sender<LayerMessage>, mpsc::Receiver<LayerMessage>),
}

impl BioNervousSystem {
    /// Pipeline: L1 -> L2 (cache) -> L3 (if miss) -> L4 -> L5
    pub async fn process(&self, input: RawInput) -> Result<NervousSystemOutput, Error> {
        let ctx = ProcessingContext::default();

        // L1: Sensing
        let sensing_out = self.sensing.process(input, &ctx).await?;

        // L2: Reflex (cache check)
        if let Some(response) = self.reflex.query(&sensing_out.embedding, &ctx).await {
            if self.reflex.should_bypass(response.confidence) {
                return Ok(NervousSystemOutput::from_cache(response));
            }
        }

        // L3: Memory
        let patterns = self.memory.retrieve(&sensing_out.embedding, 10, 0.2).await;

        // L4: Learning
        let utl_state = UTLState {
            delta_s: sensing_out.delta_s,
            delta_c: self.compute_coherence(&patterns),
            w_e: 1.0,
            phi: self.coherence.get_phase().await,
        };
        let learning_out = self.learning.process(&utl_state, &ctx).await?;

        // L5: Coherence
        let coherence_out = self.coherence.process(CoherenceInput::default(), &ctx).await?;

        // Optional: Formal verification
        let verification = self.verification.write().await.verify(&sensing_out.content).await;

        Ok(NervousSystemOutput { learning_score: learning_out.learning_score, phi: coherence_out.phi, verification })
    }
}
```

---

## 7. Performance Enforcement

| Layer | Budget | Enforcement |
|-------|--------|-------------|
| L1 Sensing | <5ms | tokio::time::timeout |
| L2 Reflex | <100us | Inline measurement |
| L3 Memory | <1ms | FAISS GPU |
| L4 Learning | <10ms | Batched 100Hz |
| L5 Coherence | <10ms | Async sync |
| E2E Pipeline | <25ms P95 | Aggregate |

---

## 8. Configuration

```toml
[nervous_system]
enabled = true
verification_enabled = true

[nervous_system.sensing]
latency_budget_ms = 5
pii_scrubbing_enabled = true

[nervous_system.reflex]
latency_budget_us = 100
confidence_threshold = 0.95

[nervous_system.memory]
latency_budget_ms = 1
initial_beta = 1.0
gpu_enabled = true

[nervous_system.learning]
latency_budget_ms = 10
update_hz = 100
gradient_clip = 1.0

[nervous_system.coherence]
latency_budget_ms = 10
sync_interval_ms = 10

[nervous_system.verification]
timeout_ms = 100
cache_size = 10000
```

---

## 9. Dependencies

```toml
[dependencies]
context-graph-core = { path = "../context-graph-core" }
context-graph-utl = { path = "../context-graph-utl" }
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"
dashmap = "5.5"
uuid = { version = "1.6", features = ["v4"] }
regex = "1.10"
lru = "0.12"
sha2 = "0.10"
thiserror = "1.0"
faiss = { version = "0.12", features = ["gpu"], optional = true }
```

---

## 10. Acceptance Criteria

- [ ] L1 Sensing: <5ms P95, PII scrubbing, adversarial detection
- [ ] L2 Reflex: <100us P95, >80% cache hit rate
- [ ] L3 Memory: <1ms P95, Modern Hopfield, FAISS GPU
- [ ] L4 Learning: <10ms P95, 100Hz, UTL equation, neuromodulation
- [ ] L5 Coherence: <10ms P95, 10ms sync, phase (phi)
- [ ] FormalVerificationLayer with SMT solver (REQ-BIONV-056/057)
- [ ] LayerMessage protocol with priority/deadlines
- [ ] NervousLayer trait for all 5 layers
- [ ] >95% latency budget compliance

---

*Document Version: 1.0.0 | Generated: 2025-12-31*
