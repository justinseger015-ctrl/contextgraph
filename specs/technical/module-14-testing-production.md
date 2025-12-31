# Module 14: Testing & Production Infrastructure

## Technical Specification v1.0.0

**Status**: Draft
**Last Updated**: 2025-12-31
**Module Dependencies**: All modules (M01-M13)

---

## 1. Overview

Module 14 provides comprehensive testing infrastructure and production deployment capabilities for the ContextGraph system, including specialized Marblestone architecture validation.

### 1.1 Design Goals

- Achieve >90% unit test coverage across all modules
- Ensure >80% integration test coverage for cross-module interactions
- Maintain <2% variance in performance benchmarks
- Support 10K concurrent session load testing
- Enable fault injection with MTTR <30s recovery
- Provide production-ready deployment configurations

---

## 2. UnitTestFramework

### 2.1 Architecture

```
UnitTestFramework
├── TestRunner (nextest-based)
├── MockFactory (dependency injection)
├── CoverageCollector (llvm-cov integration)
├── AssertionLibrary (custom matchers)
└── FixtureManager (test data generation)
```

### 2.2 Interface Definition

```rust
pub trait UnitTestFramework {
    /// Execute test suite with filtering
    fn run_tests(&self, filter: TestFilter) -> TestResults;

    /// Generate coverage report
    fn collect_coverage(&self) -> CoverageReport;

    /// Create mock for dependency injection
    fn create_mock<T: Mockable>(&self) -> Mock<T>;

    /// Load test fixtures
    fn load_fixtures(&self, path: &Path) -> Fixtures;
}

pub struct TestFilter {
    pub module: Option<String>,
    pub test_name: Option<String>,
    pub tags: Vec<String>,
    pub exclude_slow: bool,
}

pub struct TestResults {
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration: Duration,
    pub failures: Vec<TestFailure>,
}

pub struct CoverageReport {
    pub line_coverage: f64,      // Target: >90%
    pub branch_coverage: f64,    // Target: >85%
    pub function_coverage: f64,  // Target: >95%
    pub uncovered_lines: Vec<UncoveredLine>,
}
```

### 2.3 Nextest Configuration

```toml
# .config/nextest.toml
[profile.default]
retries = 2
slow-timeout = { period = "60s", terminate-after = 2 }
fail-fast = false

[profile.ci]
retries = 3
test-threads = "num-cpus"
failure-output = "immediate-final"

[profile.coverage]
retries = 0
test-threads = 1
```

### 2.4 Coverage Requirements by Module

| Module | Line Coverage | Branch Coverage | Function Coverage |
|--------|---------------|-----------------|-------------------|
| M01 Core | 95% | 90% | 98% |
| M02 Storage | 92% | 88% | 95% |
| M03 HNSW | 90% | 85% | 95% |
| M04 Temporal | 92% | 88% | 95% |
| M05 Memory | 90% | 85% | 92% |
| M06 Query | 93% | 90% | 96% |
| M07 Aggregation | 91% | 87% | 94% |
| M08 Security | 95% | 92% | 98% |
| M09 Federation | 88% | 82% | 92% |
| M10 Sync | 90% | 85% | 93% |
| M11 MCP | 92% | 88% | 95% |
| M12 Learning | 88% | 80% | 90% |
| M13 Marblestone | 90% | 85% | 93% |

---

## 3. IntegrationTestHarness

### 3.1 Architecture

```
IntegrationTestHarness
├── TestOrchestrator (scenario coordination)
├── ServiceManager (container lifecycle)
├── StateManager (database snapshots)
├── NetworkSimulator (latency/partition injection)
└── AssertionChain (multi-step validation)
```

### 3.2 Interface Definition

```rust
pub trait IntegrationTestHarness {
    /// Setup test environment
    async fn setup(&mut self, config: TestConfig) -> Result<TestEnvironment>;

    /// Execute integration scenario
    async fn run_scenario(&self, scenario: Scenario) -> ScenarioResult;

    /// Teardown and cleanup
    async fn teardown(&mut self) -> Result<()>;

    /// Inject network conditions
    fn inject_network_condition(&self, condition: NetworkCondition);
}

pub struct Scenario {
    pub name: String,
    pub steps: Vec<ScenarioStep>,
    pub assertions: Vec<Assertion>,
    pub timeout: Duration,
}

pub enum ScenarioStep {
    CreateContext(ContextSpec),
    QueryContext(QuerySpec),
    UpdateContext(UpdateSpec),
    SimulateFailure(FailureSpec),
    WaitForSync(SyncSpec),
    ValidateState(StateSpec),
}

pub struct ScenarioResult {
    pub success: bool,
    pub step_results: Vec<StepResult>,
    pub duration: Duration,
    pub resource_usage: ResourceMetrics,
}
```

### 3.3 Cross-Module Integration Tests

```rust
#[integration_test]
async fn test_context_lifecycle_full_stack() {
    let harness = IntegrationTestHarness::new();

    let scenario = Scenario::builder()
        .name("context_lifecycle")
        .step(ScenarioStep::CreateContext(ContextSpec {
            content: "Test context for integration",
            metadata: json!({"source": "integration_test"}),
        }))
        .step(ScenarioStep::QueryContext(QuerySpec {
            query: "test context",
            expected_results: 1,
        }))
        .step(ScenarioStep::UpdateContext(UpdateSpec {
            content_delta: "Updated content",
        }))
        .step(ScenarioStep::ValidateState(StateSpec {
            expected_version: 2,
        }))
        .timeout(Duration::from_secs(30))
        .build();

    let result = harness.run_scenario(scenario).await;
    assert!(result.success);
}
```

### 3.4 Coverage Matrix

| Integration Path | Test Count | Coverage |
|-----------------|------------|----------|
| M01 <-> M02 (Core-Storage) | 24 | 92% |
| M01 <-> M03 (Core-HNSW) | 18 | 88% |
| M02 <-> M04 (Storage-Temporal) | 15 | 85% |
| M06 <-> M03 (Query-HNSW) | 22 | 90% |
| M08 <-> M11 (Security-MCP) | 20 | 95% |
| M09 <-> M10 (Federation-Sync) | 16 | 82% |
| M12 <-> M13 (Learning-Marblestone) | 12 | 85% |

---

## 4. PerformanceBenchmarks

### 4.1 Architecture

```
PerformanceBenchmarks
├── BenchmarkRunner (criterion-based)
├── MetricsCollector (statistical analysis)
├── RegressionDetector (<2% variance threshold)
├── BaselineManager (historical comparisons)
└── ReportGenerator (CI/CD integration)
```

### 4.2 Criterion Configuration

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn context_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_creation");
    group.significance_level(0.01);  // 1% significance
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter(|| create_contexts(size));
            },
        );
    }
    group.finish();
}

fn hnsw_search_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    group.significance_level(0.02);  // 2% variance threshold

    for k in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("top_k", k),
            k,
            |b, &k| {
                b.iter(|| hnsw_search(query_vector(), k));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, context_creation_benchmark, hnsw_search_benchmark);
criterion_main!(benches);
```

### 4.3 Performance Baselines

| Benchmark | Baseline | Variance | Regression Threshold |
|-----------|----------|----------|---------------------|
| Context Creation (1K) | 15ms | 1.2% | +5% |
| HNSW Search (top-100) | 2.5ms | 1.8% | +3% |
| Query Execution | 8ms | 1.5% | +5% |
| Temporal Range Query | 12ms | 1.9% | +5% |
| Memory Consolidation | 45ms | 1.7% | +10% |
| Federation Sync | 120ms | 2.0% | +10% |
| Marblestone Forward Pass | 5ms | 1.5% | +3% |

---

## 5. LoadTester

### 5.1 Architecture

```
LoadTester
├── SessionGenerator (10K concurrent)
├── WorkloadProfiles (realistic patterns)
├── MetricsAggregator (p50/p95/p99)
├── ResourceMonitor (CPU/memory/network)
└── BreakpointDetector (capacity limits)
```

### 5.2 Interface Definition

```rust
pub trait LoadTester {
    /// Configure load test parameters
    fn configure(&mut self, config: LoadConfig) -> Result<()>;

    /// Execute load test
    async fn run(&self) -> LoadTestResult;

    /// Get real-time metrics
    fn get_metrics(&self) -> LiveMetrics;

    /// Find system breakpoint
    async fn find_breakpoint(&self) -> BreakpointResult;
}

pub struct LoadConfig {
    pub concurrent_sessions: usize,    // Target: 10K
    pub ramp_up_duration: Duration,
    pub steady_state_duration: Duration,
    pub workload_profile: WorkloadProfile,
    pub target_throughput: Option<f64>,
}

pub struct WorkloadProfile {
    pub read_ratio: f64,               // 70%
    pub write_ratio: f64,              // 20%
    pub query_ratio: f64,              // 10%
    pub think_time: Distribution,
}

pub struct LoadTestResult {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub resource_usage: ResourceUsage,
}
```

### 5.3 Load Test Scenarios

```yaml
# load_tests/scenarios/standard.yaml
scenarios:
  - name: "10K_concurrent_sessions"
    concurrent_sessions: 10000
    ramp_up: "2m"
    steady_state: "10m"
    workload:
      read: 70%
      write: 20%
      query: 10%
    thresholds:
      p95_latency: "100ms"
      error_rate: "0.1%"
      throughput: "5000 req/s"

  - name: "burst_traffic"
    concurrent_sessions: 15000
    ramp_up: "30s"
    steady_state: "5m"
    workload:
      read: 80%
      write: 15%
      query: 5%
    thresholds:
      p99_latency: "500ms"
      error_rate: "1%"

  - name: "sustained_write"
    concurrent_sessions: 5000
    ramp_up: "1m"
    steady_state: "30m"
    workload:
      read: 30%
      write: 60%
      query: 10%
    thresholds:
      p95_latency: "200ms"
      error_rate: "0.5%"
```

---

## 6. ChaosEngine

### 6.1 Architecture

```
ChaosEngine
├── FaultInjector (controlled failures)
├── RecoveryMonitor (MTTR tracking)
├── ImpactAnalyzer (blast radius)
├── GameDayOrchestrator (scheduled chaos)
└── SafetyController (abort conditions)
```

### 6.2 Interface Definition

```rust
pub trait ChaosEngine {
    /// Inject fault into system
    async fn inject_fault(&self, fault: FaultSpec) -> FaultHandle;

    /// Monitor recovery
    async fn monitor_recovery(&self, handle: FaultHandle) -> RecoveryMetrics;

    /// Abort fault injection
    async fn abort(&self, handle: FaultHandle) -> Result<()>;

    /// Schedule game day
    fn schedule_gameday(&self, schedule: GameDaySchedule) -> GameDayHandle;
}

pub enum FaultSpec {
    NodeFailure { node_id: String, duration: Duration },
    NetworkPartition { partition: Vec<Vec<String>>, duration: Duration },
    LatencySpike { target: String, latency: Duration, duration: Duration },
    DiskFailure { node_id: String, disk_id: String },
    MemoryPressure { node_id: String, pressure_mb: usize },
    CpuStarvation { node_id: String, utilization: f64 },
    ProcessCrash { process: String, restart_after: Duration },
}

pub struct RecoveryMetrics {
    pub time_to_detect: Duration,
    pub time_to_recover: Duration,      // Target: <30s
    pub data_loss: Option<usize>,
    pub service_degradation: f64,
    pub automatic_recovery: bool,
}
```

### 6.3 Chaos Scenarios

```yaml
# chaos/scenarios/standard.yaml
scenarios:
  - name: "single_node_failure"
    fault:
      type: "node_failure"
      target: "random"
      duration: "5m"
    expected_recovery:
      mttr: "30s"
      data_loss: 0
      automatic: true

  - name: "network_partition"
    fault:
      type: "network_partition"
      partitions:
        - ["node-1", "node-2"]
        - ["node-3", "node-4", "node-5"]
      duration: "2m"
    expected_recovery:
      mttr: "15s"
      consistency: "eventual"

  - name: "cascading_failure"
    faults:
      - type: "cpu_starvation"
        target: "node-1"
        utilization: 95%
        at: "0s"
      - type: "memory_pressure"
        target: "node-2"
        pressure_mb: 8192
        at: "30s"
    expected_recovery:
      mttr: "45s"
      graceful_degradation: true
```

---

## 7. ProductionConfig

### 7.1 Docker Configuration

```dockerfile
# Dockerfile.production
FROM rust:1.75-slim AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --features production

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/contextgraph /usr/local/bin/

RUN groupadd -r contextgraph && useradd -r -g contextgraph contextgraph
USER contextgraph

EXPOSE 8080 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["contextgraph"]
CMD ["--config", "/etc/contextgraph/config.toml"]
```

### 7.2 Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: contextgraph
  labels:
    app: contextgraph
spec:
  replicas: 3
  selector:
    matchLabels:
      app: contextgraph
  template:
    metadata:
      labels:
        app: contextgraph
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: contextgraph
      containers:
        - name: contextgraph
          image: contextgraph:latest
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          env:
            - name: RUST_LOG
              value: "info,contextgraph=debug"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: contextgraph-secrets
                  key: database-url
          volumeMounts:
            - name: config
              mountPath: /etc/contextgraph
      volumes:
        - name: config
          configMap:
            name: contextgraph-config
---
apiVersion: v1
kind: Service
metadata:
  name: contextgraph
spec:
  selector:
    app: contextgraph
  ports:
    - port: 80
      targetPort: 8080
      name: http
    - port: 9090
      targetPort: 9090
      name: metrics
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: contextgraph
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: contextgraph
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

---

## 8. MonitoringStack

### 8.1 Prometheus Metrics

```rust
use prometheus::{Registry, Counter, Histogram, Gauge};

pub struct ContextGraphMetrics {
    // Request metrics
    pub requests_total: Counter,
    pub request_duration: Histogram,
    pub active_connections: Gauge,

    // Context metrics
    pub contexts_created: Counter,
    pub contexts_queried: Counter,
    pub context_storage_bytes: Gauge,

    // HNSW metrics
    pub hnsw_search_duration: Histogram,
    pub hnsw_index_size: Gauge,

    // Marblestone metrics
    pub marblestone_forward_duration: Histogram,
    pub marblestone_active_modules: Gauge,
    pub marblestone_memory_capacity: Gauge,

    // Error metrics
    pub errors_total: Counter,
    pub recovery_time: Histogram,
}

impl ContextGraphMetrics {
    pub fn register(registry: &Registry) -> Self {
        Self {
            requests_total: Counter::new(
                "contextgraph_requests_total",
                "Total number of requests"
            ).unwrap(),
            request_duration: Histogram::with_opts(
                HistogramOpts::new(
                    "contextgraph_request_duration_seconds",
                    "Request duration in seconds"
                ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
            ).unwrap(),
            // ... additional metrics
        }
    }
}
```

### 8.2 Alerting Rules

```yaml
# prometheus/alerts.yaml
groups:
  - name: contextgraph
    rules:
      - alert: HighErrorRate
        expr: rate(contextgraph_errors_total[5m]) > 0.01
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(contextgraph_request_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          description: "P95 latency is {{ $value | humanizeDuration }}"

      - alert: LowRecoveryTime
        expr: histogram_quantile(0.99, contextgraph_recovery_time_seconds) > 30
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Recovery time exceeds 30s threshold"
          description: "MTTR is {{ $value | humanizeDuration }}"

      - alert: MarblestoneModuleFailure
        expr: contextgraph_marblestone_active_modules < 3
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Marblestone module count below minimum"
```

---

## 9. Marblestone Test Cases (T-MARB-001 to T-MARB-008)

### 9.1 Test Case Definitions

```rust
/// T-MARB-001: ModularNN Forward Pass Validation
#[test]
fn t_marb_001_modular_nn_forward_pass() {
    let network = ModularNeuralNetwork::new(ModularNNConfig::default());
    let input = Tensor::random(&[32, 768]);

    let output = network.forward(&input);

    assert_eq!(output.shape(), &[32, 768]);
    assert!(output.values().iter().all(|v| v.is_finite()));
    assert!(network.get_active_modules() >= 3);
}

/// T-MARB-002: Routing Network Decision Quality
#[test]
fn t_marb_002_routing_network_decisions() {
    let router = RoutingNetwork::new(RoutingConfig::default());
    let context = ContextVector::random(768);

    let routes = router.compute_routes(&context);

    assert!(routes.len() >= 2);
    assert!(routes.iter().map(|r| r.weight).sum::<f32>() - 1.0 < 0.01);
    assert!(routes[0].confidence > 0.7);
}

/// T-MARB-003: Working Memory Capacity Test
#[test]
fn t_marb_003_working_memory_capacity() {
    let mut memory = WorkingMemory::new(WorkingMemoryConfig {
        capacity: 1000,
        consolidation_threshold: 0.8,
    });

    // Fill to capacity
    for i in 0..1000 {
        memory.store(format!("item_{}", i), Tensor::random(&[768]));
    }

    assert_eq!(memory.len(), 1000);
    assert!(memory.utilization() > 0.95);

    // Test overflow handling
    memory.store("overflow_item", Tensor::random(&[768]));
    assert_eq!(memory.len(), 1000); // Should consolidate, not exceed
}

/// T-MARB-004: Attention Mechanism Coherence
#[test]
fn t_marb_004_attention_coherence() {
    let attention = MarblestoneAttention::new(AttentionConfig::default());
    let query = Tensor::random(&[8, 16, 768]);
    let key = Tensor::random(&[8, 32, 768]);
    let value = Tensor::random(&[8, 32, 768]);

    let (output, weights) = attention.forward(query, key, value);

    assert_eq!(output.shape(), &[8, 16, 768]);
    assert!(weights.sum_axis(2).values().iter().all(|v| (*v - 1.0).abs() < 0.01));
}

/// T-MARB-005: Module Specialization Verification
#[test]
fn t_marb_005_module_specialization() {
    let mut network = ModularNeuralNetwork::new(ModularNNConfig::default());

    // Train with domain-specific data
    let spatial_data = generate_spatial_data(1000);
    let temporal_data = generate_temporal_data(1000);

    network.train_specialized(&spatial_data, ModuleType::Spatial);
    network.train_specialized(&temporal_data, ModuleType::Temporal);

    let spatial_module = network.get_module(ModuleType::Spatial);
    let temporal_module = network.get_module(ModuleType::Temporal);

    // Verify specialization divergence
    let similarity = cosine_similarity(
        &spatial_module.weights(),
        &temporal_module.weights()
    );
    assert!(similarity < 0.5); // Should be specialized, not similar
}

/// T-MARB-006: Consolidation Pipeline Integrity
#[test]
fn t_marb_006_consolidation_pipeline() {
    let mut pipeline = ConsolidationPipeline::new(ConsolidationConfig {
        batch_size: 64,
        compression_ratio: 0.5,
    });

    let memories: Vec<Memory> = (0..1000)
        .map(|i| Memory::new(format!("mem_{}", i), Tensor::random(&[768])))
        .collect();

    let consolidated = pipeline.consolidate(&memories);

    assert!(consolidated.len() <= 500); // 50% compression
    assert!(pipeline.information_retained() > 0.9); // 90% info retention
}

/// T-MARB-007: Gating Mechanism Stability
#[test]
fn t_marb_007_gating_stability() {
    let gating = GatingMechanism::new(GatingConfig::default());
    let inputs: Vec<Tensor> = (0..100)
        .map(|_| Tensor::random(&[768]))
        .collect();

    let mut gate_variance = Vec::new();
    for input in &inputs {
        let gates = gating.compute_gates(input);
        gate_variance.push(gates.variance());
    }

    let avg_variance = gate_variance.iter().sum::<f32>() / gate_variance.len() as f32;
    assert!(avg_variance < 0.3); // Gates should be stable
}

/// T-MARB-008: Hierarchical Processing Correctness
#[test]
fn t_marb_008_hierarchical_processing() {
    let hierarchy = HierarchicalProcessor::new(HierarchyConfig {
        levels: 4,
        pooling: PoolingStrategy::Attention,
    });

    let input = Tensor::random(&[1, 256, 768]); // Sequence of 256 tokens

    let outputs = hierarchy.process_all_levels(&input);

    assert_eq!(outputs.len(), 4);
    assert_eq!(outputs[0].shape(), &[1, 256, 768]); // Level 0: full resolution
    assert_eq!(outputs[1].shape(), &[1, 64, 768]);  // Level 1: 4x pooled
    assert_eq!(outputs[2].shape(), &[1, 16, 768]);  // Level 2: 16x pooled
    assert_eq!(outputs[3].shape(), &[1, 4, 768]);   // Level 3: 64x pooled
}
```

---

## 10. Marblestone Integration Tests

### 10.1 Integration Test Suite

```rust
/// MI-001: Marblestone-HNSW Integration
#[integration_test]
async fn mi_001_marblestone_hnsw_integration() {
    let harness = IntegrationTestHarness::new();
    let marblestone = MarblestoneEngine::new(Default::default());
    let hnsw = HnswIndex::new(Default::default());

    // Generate embeddings through Marblestone
    let contexts: Vec<Context> = generate_test_contexts(1000);
    let embeddings: Vec<Tensor> = contexts.iter()
        .map(|c| marblestone.encode(c))
        .collect();

    // Index in HNSW
    for (i, emb) in embeddings.iter().enumerate() {
        hnsw.insert(i as u64, emb.as_slice());
    }

    // Query through Marblestone
    let query_context = Context::new("search query");
    let query_embedding = marblestone.encode(&query_context);
    let results = hnsw.search(query_embedding.as_slice(), 10);

    assert_eq!(results.len(), 10);
    assert!(results[0].distance < 0.5);
}

/// MI-002: Marblestone-Temporal Integration
#[integration_test]
async fn mi_002_marblestone_temporal_integration() {
    let marblestone = MarblestoneEngine::new(Default::default());
    let temporal = TemporalIndex::new(Default::default());

    // Create time-aware contexts
    let now = Timestamp::now();
    for i in 0..100 {
        let context = Context::new(format!("Event at time {}", i))
            .with_timestamp(now - Duration::hours(i));
        let embedding = marblestone.encode(&context);
        temporal.insert(context.timestamp, embedding);
    }

    // Query temporal range
    let range_start = now - Duration::hours(50);
    let range_end = now - Duration::hours(10);
    let results = temporal.query_range(range_start, range_end);

    assert_eq!(results.len(), 40);
}

/// MI-003: Marblestone-Memory Consolidation Integration
#[integration_test]
async fn mi_003_marblestone_memory_consolidation() {
    let marblestone = MarblestoneEngine::new(Default::default());
    let memory_system = MemorySystem::new(Default::default());

    // Store many short-term memories
    for i in 0..500 {
        let context = Context::new(format!("Short-term memory {}", i));
        let embedding = marblestone.encode(&context);
        memory_system.store_short_term(embedding);
    }

    // Trigger consolidation
    memory_system.consolidate();

    // Verify long-term memory formation
    assert!(memory_system.long_term_count() > 0);
    assert!(memory_system.long_term_count() < 500); // Should be compressed

    // Verify retrieval still works
    let query = Context::new("memory query");
    let query_emb = marblestone.encode(&query);
    let retrieved = memory_system.retrieve(&query_emb, 10);
    assert!(!retrieved.is_empty());
}

/// MI-004: Marblestone-Query Pipeline Integration
#[integration_test]
async fn mi_004_marblestone_query_pipeline() {
    let engine = ContextGraphEngine::new(Default::default());

    // Setup with Marblestone encoding
    engine.enable_marblestone(MarblestoneConfig::default());

    // Insert contexts
    for i in 0..1000 {
        engine.insert(Context::new(format!("Document {} about topic X", i)));
    }

    // Execute complex query
    let query = Query::builder()
        .semantic("topic X")
        .temporal(TimeRange::last_days(7))
        .limit(50)
        .build();

    let results = engine.execute(query).await;

    assert_eq!(results.len(), 50);
    assert!(results.iter().all(|r| r.relevance_score > 0.5));
}

/// MI-005: Marblestone-Federation Sync Integration
#[integration_test]
async fn mi_005_marblestone_federation_sync() {
    let node_a = FederatedNode::new("node-a", MarblestoneConfig::default());
    let node_b = FederatedNode::new("node-b", MarblestoneConfig::default());

    // Insert on node A
    for i in 0..100 {
        node_a.insert(Context::new(format!("Context from A: {}", i)));
    }

    // Sync to node B
    node_a.sync_to(&node_b).await;

    // Query on node B should find contexts
    let results = node_b.query("Context from A").await;
    assert!(!results.is_empty());

    // Verify embedding consistency
    let emb_a = node_a.get_embedding("Context from A: 0");
    let emb_b = node_b.get_embedding("Context from A: 0");
    assert!(cosine_similarity(&emb_a, &emb_b) > 0.99);
}

/// MI-006: Marblestone-Security Integration
#[integration_test]
async fn mi_006_marblestone_security_integration() {
    let engine = ContextGraphEngine::new(Default::default());
    engine.enable_marblestone(MarblestoneConfig::default());
    engine.enable_security(SecurityConfig::default());

    // Create contexts with different security levels
    let public_ctx = Context::new("Public information")
        .with_security_level(SecurityLevel::Public);
    let private_ctx = Context::new("Private information")
        .with_security_level(SecurityLevel::Private);

    engine.insert(public_ctx);
    engine.insert(private_ctx);

    // Query as public user
    let public_user = User::new("public").with_clearance(SecurityLevel::Public);
    let results = engine.query_as("information", &public_user).await;

    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("Public"));
}
```

---

## 11. Marblestone Benchmarks

### 11.1 Benchmark Suite

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

/// B-MARB-001: Forward Pass Throughput
fn b_marb_001_forward_throughput(c: &mut Criterion) {
    let network = ModularNeuralNetwork::new(ModularNNConfig::default());

    let mut group = c.benchmark_group("marblestone_forward");
    for batch_size in [1, 8, 32, 64, 128].iter() {
        let input = Tensor::random(&[*batch_size, 768]);
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &input,
            |b, input| b.iter(|| network.forward(input)),
        );
    }
    group.finish();
}

/// B-MARB-002: Routing Latency
fn b_marb_002_routing_latency(c: &mut Criterion) {
    let router = RoutingNetwork::new(RoutingConfig::default());
    let context = ContextVector::random(768);

    c.bench_function("routing_latency", |b| {
        b.iter(|| router.compute_routes(&context))
    });
}

/// B-MARB-003: Memory Access Patterns
fn b_marb_003_memory_access(c: &mut Criterion) {
    let mut memory = WorkingMemory::new(WorkingMemoryConfig::default());

    // Pre-fill memory
    for i in 0..1000 {
        memory.store(format!("key_{}", i), Tensor::random(&[768]));
    }

    let mut group = c.benchmark_group("memory_access");

    group.bench_function("read", |b| {
        b.iter(|| memory.retrieve("key_500"))
    });

    group.bench_function("write", |b| {
        let value = Tensor::random(&[768]);
        b.iter(|| memory.store("new_key", value.clone()))
    });

    group.bench_function("search", |b| {
        let query = Tensor::random(&[768]);
        b.iter(|| memory.search(&query, 10))
    });

    group.finish();
}

/// B-MARB-004: Attention Computation
fn b_marb_004_attention_computation(c: &mut Criterion) {
    let attention = MarblestoneAttention::new(AttentionConfig::default());

    let mut group = c.benchmark_group("attention");
    for seq_len in [64, 128, 256, 512].iter() {
        let q = Tensor::random(&[8, *seq_len, 768]);
        let k = Tensor::random(&[8, *seq_len, 768]);
        let v = Tensor::random(&[8, *seq_len, 768]);

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &(q.clone(), k.clone(), v.clone()),
            |b, (q, k, v)| b.iter(|| attention.forward(q.clone(), k.clone(), v.clone())),
        );
    }
    group.finish();
}

/// B-MARB-005: Consolidation Throughput
fn b_marb_005_consolidation_throughput(c: &mut Criterion) {
    let pipeline = ConsolidationPipeline::new(ConsolidationConfig::default());

    let mut group = c.benchmark_group("consolidation");
    for memory_count in [100, 500, 1000, 5000].iter() {
        let memories: Vec<Memory> = (0..*memory_count)
            .map(|i| Memory::new(format!("m_{}", i), Tensor::random(&[768])))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(memory_count),
            &memories,
            |b, memories| b.iter(|| pipeline.consolidate(memories)),
        );
    }
    group.finish();
}

/// B-MARB-006: Module Switching Overhead
fn b_marb_006_module_switching(c: &mut Criterion) {
    let mut network = ModularNeuralNetwork::new(ModularNNConfig::default());

    c.bench_function("module_switch", |b| {
        b.iter(|| {
            network.activate_module(ModuleType::Spatial);
            network.activate_module(ModuleType::Temporal);
            network.activate_module(ModuleType::Semantic);
        })
    });
}

/// B-MARB-007: End-to-End Encoding
fn b_marb_007_e2e_encoding(c: &mut Criterion) {
    let engine = MarblestoneEngine::new(MarblestoneConfig::default());

    let mut group = c.benchmark_group("e2e_encoding");
    for text_len in [100, 500, 1000, 2000].iter() {
        let text = generate_text(*text_len);
        let context = Context::new(text);

        group.bench_with_input(
            BenchmarkId::new("chars", text_len),
            &context,
            |b, ctx| b.iter(|| engine.encode(ctx)),
        );
    }
    group.finish();
}

criterion_group!(
    marblestone_benches,
    b_marb_001_forward_throughput,
    b_marb_002_routing_latency,
    b_marb_003_memory_access,
    b_marb_004_attention_computation,
    b_marb_005_consolidation_throughput,
    b_marb_006_module_switching,
    b_marb_007_e2e_encoding
);
criterion_main!(marblestone_benches);
```

### 11.2 Benchmark Baselines

| Benchmark | Baseline | Variance | Target |
|-----------|----------|----------|--------|
| B-MARB-001 (batch=32) | 5.2ms | 1.3% | <8ms |
| B-MARB-002 | 0.8ms | 1.1% | <1ms |
| B-MARB-003 (read) | 0.05ms | 0.8% | <0.1ms |
| B-MARB-003 (write) | 0.12ms | 1.2% | <0.2ms |
| B-MARB-003 (search) | 2.1ms | 1.5% | <3ms |
| B-MARB-004 (seq=256) | 8.5ms | 1.8% | <12ms |
| B-MARB-005 (n=1000) | 45ms | 1.7% | <60ms |
| B-MARB-006 | 0.3ms | 0.9% | <0.5ms |
| B-MARB-007 (1000 chars) | 12ms | 1.4% | <15ms |

---

## 12. Requirements Coverage Matrix

### 12.1 REQ-*-0XX Coverage

| Requirement | Test Coverage | Integration Coverage |
|-------------|---------------|---------------------|
| REQ-MARB-001 (ModularNN) | T-MARB-001, T-MARB-005 | MI-001, MI-004 |
| REQ-MARB-002 (Routing) | T-MARB-002 | MI-004 |
| REQ-MARB-003 (WorkingMemory) | T-MARB-003 | MI-003 |
| REQ-MARB-004 (Attention) | T-MARB-004 | MI-001, MI-004 |
| REQ-MARB-005 (Consolidation) | T-MARB-006 | MI-003 |
| REQ-MARB-006 (Gating) | T-MARB-007 | MI-004 |
| REQ-MARB-007 (Hierarchy) | T-MARB-008 | MI-002 |
| REQ-SEC-001 (Auth) | T-SEC-* | MI-006 |
| REQ-FED-001 (Sync) | T-FED-* | MI-005 |

---

## 13. CI/CD Integration

### 13.1 GitHub Actions Workflow

```yaml
# .github/workflows/test.yaml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: taiki-e/install-action@nextest

      - name: Run unit tests
        run: cargo nextest run --profile ci

      - name: Generate coverage
        run: cargo llvm-cov nextest --lcov --output-path lcov.info

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: lcov.info
          fail_ci_if_error: true

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: cargo test --test '*_integration' --features integration

  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmarks
        run: cargo bench --bench '*' -- --noplot
      - name: Check regression
        run: ./scripts/check_benchmark_regression.sh

  load-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Run load tests
        run: ./scripts/run_load_tests.sh
```

---

## 14. Summary

Module 14 provides comprehensive testing and production infrastructure:

- **Unit Testing**: >90% coverage with nextest
- **Integration Testing**: >80% coverage across module boundaries
- **Performance Benchmarks**: <2% variance with criterion
- **Load Testing**: 10K concurrent session support
- **Chaos Engineering**: MTTR <30s fault recovery
- **Production Deployment**: Docker + Kubernetes manifests
- **Monitoring**: Prometheus metrics and alerting

### Marblestone-Specific Coverage:
- 8 dedicated test cases (T-MARB-001 to T-MARB-008)
- 6 integration tests covering cross-module interactions
- 7 performance benchmarks with regression detection
- Full coverage of REQ-MARB-0XX requirements

---

*Document generated for ContextGraph v1.0.0*
