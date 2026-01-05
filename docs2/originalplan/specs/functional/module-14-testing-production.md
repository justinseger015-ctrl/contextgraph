# Module 14: Testing & Production - Functional Specification

**Module ID**: SPEC-TEST-014
**Version**: 1.0.0
**Status**: Draft
**Phase**: 13 (Final)
**Duration**: 3-4 weeks
**Dependencies**: Modules 1-13 (All Previous Modules)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The Testing & Production module provides comprehensive quality assurance infrastructure and production deployment capabilities for the Ultimate Context Graph system. This module establishes the testing framework, performance benchmarking suite, chaos engineering capabilities, and production-ready deployment configurations that ensure the system meets its reliability, performance, and operational requirements.

### 1.1 Core Objectives

- Establish unit testing framework with >90% code coverage across all modules
- Implement integration testing harness for cross-module validation
- Deploy performance benchmarking infrastructure using criterion
- Enable chaos engineering for resilience validation
- Provide production deployment configurations (Docker, Kubernetes)
- Configure monitoring, alerting, and observability infrastructure

### 1.2 Key Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Unit Test Coverage | >90% | cargo-llvm-cov/tarpaulin |
| Integration Test Coverage | >80% | Cross-module path coverage |
| Unit Test Execution | <5 minutes | cargo nextest total runtime |
| Integration Test Execution | <15 minutes | Full integration suite |
| Benchmark Variance | <2% | Criterion statistical analysis |
| Load Test Capacity | 10,000 concurrent | Stress test peak throughput |
| Mean Time to Recovery | <30 seconds | Chaos test recovery timing |
| Production Startup Time | <10 seconds | Container initialization |

### 1.3 Test Strategy Matrix

| Test Type | Scope | Tools | Frequency |
|-----------|-------|-------|-----------|
| Unit Tests | Single function/struct | cargo nextest, mockall | Every commit |
| Integration Tests | Cross-module interfaces | Custom harness | Every PR |
| Performance Benchmarks | Hot paths, critical operations | criterion | Weekly/release |
| Load Tests | System capacity | Custom load tester | Pre-release |
| Chaos Tests | Failure resilience | ChaosEngine | Pre-release |
| Security Tests | Attack surface | cargo-audit, fuzzing | Weekly |

---

## 2. Module Dependencies

### 2.1 Dependencies on Previous Modules

```
Module 14 Testing & Production
    |
    +-- Module 1 (Ghost System)
    |       - Test infrastructure stubs
    |       - CI/CD pipeline foundation
    |       - Workspace structure validation
    |
    +-- Module 2 (Core Infrastructure)
    |       - MemoryNode serialization tests
    |       - RocksDB storage layer tests
    |       - MCP tool contract tests
    |
    +-- Module 3 (Embedding Pipeline)
    |       - Embedding accuracy benchmarks
    |       - Batch processing load tests
    |       - Model switching tests
    |
    +-- Module 4 (Knowledge Graph)
    |       - Graph traversal benchmarks
    |       - Query optimization tests
    |       - Relationship integrity tests
    |
    +-- Module 5 (UTL Integration)
    |       - UTL processor validation
    |       - Compression ratio tests
    |       - Protocol compliance tests
    |
    +-- Module 6 (Bio-Nervous System)
    |       - Neural layer integration tests
    |       - Signal propagation benchmarks
    |       - Homeostasis validation
    |
    +-- Module 7 (CUDA Optimization)
    |       - GPU acceleration benchmarks
    |       - Memory transfer tests
    |       - Fallback behavior tests
    |
    +-- Module 8 (GPU-Direct Storage)
    |       - Direct I/O benchmarks
    |       - Storage latency tests
    |       - Cache coherence tests
    |
    +-- Module 9 (Dream Layer)
    |       - Consolidation process tests
    |       - Background processing tests
    |       - Memory promotion validation
    |
    +-- Module 10 (Neuromodulation)
    |       - Modulator response tests
    |       - Priority adjustment benchmarks
    |       - Stress response validation
    |
    +-- Module 11 (Immune System)
    |       - Threat detection tests
    |       - Quarantine mechanism tests
    |       - False positive rate benchmarks
    |
    +-- Module 12 (Active Inference)
    |       - Prediction accuracy tests
    |       - Learning rate benchmarks
    |       - Model adaptation tests
    |
    +-- Module 13 (MCP Hardening)
            - Security boundary tests
            - Rate limiting validation
            - Attack simulation tests
```

### 2.2 Cross-Module Test Coverage Requirements

| Module Pair | Integration Points | Required Coverage |
|-------------|-------------------|-------------------|
| Core <-> Embedding | Vector storage, similarity search | 95% |
| Core <-> Graph | Node relationships, traversal | 90% |
| Embedding <-> CUDA | GPU acceleration paths | 85% |
| Bio-Nervous <-> Neuromodulation | Signal processing chain | 90% |
| Immune <-> MCP Hardening | Security enforcement | 95% |
| Dream <-> Core | Memory consolidation | 85% |
| All <-> MCP Interface | Tool request/response | 100% |
| Core <-> Neuromodulation | Neurotransmitter weights, steering | 90% |
| Active Inference <-> Neuromodulation | Dopamine feedback, OmniInference | 90% |
| Graph <-> Dream | Amortized shortcut creation | 85% |
| Immune <-> MCP Hardening | Formal verification layer | 90% |
| Bio-Nervous <-> Dream | Lifecycle lambda weight propagation | 85% |

---

## 3. Functional Requirements

### REQ-TEST-001: Unit Test Framework

**Priority**: Critical
**Category**: Testing Infrastructure
**Description**: The system SHALL provide a comprehensive unit testing framework that supports isolated testing of all components with mocking capabilities and coverage reporting.

#### 3.1.1 UnitTestFramework Structure

```rust
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Unit test framework configuration and execution engine.
///
/// Provides test organization, execution, mocking support, and coverage
/// collection for all system components.
///
/// Constraint: Total unit test execution < 5 minutes
/// Constraint: Code coverage > 90%
pub struct UnitTestFramework {
    /// Test discovery configuration
    pub discovery: TestDiscoveryConfig,

    /// Mock registry for dependency injection
    pub mock_registry: MockRegistry,

    /// Coverage collector configuration
    pub coverage: CoverageConfig,

    /// Test execution settings
    pub execution: ExecutionConfig,

    /// Test result aggregator
    pub results: TestResultAggregator,

    /// Flaky test detector
    pub flaky_detector: FlakyTestDetector,
}

/// Test discovery configuration
#[derive(Clone, Debug)]
pub struct TestDiscoveryConfig {
    /// Root paths to scan for tests
    pub test_roots: Vec<PathBuf>,

    /// Test file patterns (e.g., "*_test.rs", "test_*.rs")
    pub file_patterns: Vec<String>,

    /// Module patterns to include
    pub include_patterns: Vec<String>,

    /// Module patterns to exclude
    pub exclude_patterns: Vec<String>,

    /// Maximum test discovery depth
    pub max_depth: usize,
}

/// Mock registry for dependency management
pub struct MockRegistry {
    /// Registered mock implementations
    mocks: HashMap<String, Box<dyn MockProvider>>,

    /// Mock call recordings for verification
    call_records: HashMap<String, Vec<MockCall>>,

    /// Expected call sequences
    expectations: HashMap<String, Vec<MockExpectation>>,
}

/// Coverage collection configuration
#[derive(Clone, Debug)]
pub struct CoverageConfig {
    /// Coverage tool (llvm-cov, tarpaulin)
    pub tool: CoverageTool,

    /// Minimum line coverage threshold
    pub min_line_coverage: f64,  // Default: 90.0

    /// Minimum branch coverage threshold
    pub min_branch_coverage: f64,  // Default: 80.0

    /// Paths to include in coverage
    pub include_paths: Vec<PathBuf>,

    /// Paths to exclude from coverage
    pub exclude_paths: Vec<PathBuf>,

    /// Output format (lcov, html, json)
    pub output_format: CoverageFormat,

    /// Coverage report output path
    pub output_path: PathBuf,
}

#[derive(Clone, Debug)]
pub enum CoverageTool {
    LlvmCov,
    Tarpaulin,
}

/// Execution configuration
#[derive(Clone, Debug)]
pub struct ExecutionConfig {
    /// Maximum parallel test threads
    pub max_threads: usize,  // Default: num_cpus

    /// Individual test timeout
    pub test_timeout: Duration,  // Default: 30s

    /// Total suite timeout
    pub suite_timeout: Duration,  // Default: 5min

    /// Retry count for flaky tests
    pub retry_count: usize,  // Default: 0 (no retries)

    /// Fail fast on first error
    pub fail_fast: bool,  // Default: false

    /// Test execution order
    pub order: TestOrder,
}

#[derive(Clone, Debug)]
pub enum TestOrder {
    Alphabetical,
    Random { seed: u64 },
    Dependency,
}

/// Flaky test detection
pub struct FlakyTestDetector {
    /// Historical test results
    history: HashMap<String, Vec<TestOutcome>>,

    /// Flakiness threshold (failure rate to flag)
    threshold: f64,  // Default: 0.01 (1%)

    /// Minimum runs for statistical significance
    min_runs: usize,  // Default: 10
}
```

#### 3.1.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-001-01 | Unit tests execute in parallel with configurable thread count | Run with `cargo nextest run --jobs N` |
| AC-TEST-001-02 | Coverage report shows >90% line coverage | Coverage tool output verification |
| AC-TEST-001-03 | Mock registry allows dependency injection for all traits | Unit test isolation verification |
| AC-TEST-001-04 | Flaky test detector flags tests with >1% failure rate | Statistical analysis of test runs |
| AC-TEST-001-05 | Total unit test suite completes in <5 minutes | CI timing measurement |
| AC-TEST-001-06 | Test failures include stack traces and context | Failure output inspection |

#### 3.1.3 Verification Method

```bash
# Run unit tests with coverage
cargo llvm-cov nextest --all-features --workspace \
    --ignore-filename-regex 'tests/' \
    --fail-under-lines 90 \
    --fail-under-branches 80 \
    --lcov --output-path coverage.lcov

# Verify execution time
time cargo nextest run --workspace --status-level all
# Expected: real < 5m00s
```

---

### REQ-TEST-002: Integration Test Harness

**Priority**: Critical
**Category**: Testing Infrastructure
**Description**: The system SHALL provide an integration test harness that validates cross-module interactions, manages test fixtures, and ensures proper cleanup between tests.

#### 3.2.1 IntegrationTestHarness Structure

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Integration test harness for cross-module validation.
///
/// Manages test fixtures, provides module orchestration,
/// and ensures isolation between test scenarios.
///
/// Constraint: Integration test execution < 15 minutes
/// Constraint: Cross-module coverage > 80%
pub struct IntegrationTestHarness {
    /// Test fixture manager
    pub fixtures: FixtureManager,

    /// Module orchestrator for multi-component tests
    pub orchestrator: ModuleOrchestrator,

    /// Test isolation manager
    pub isolation: IsolationManager,

    /// Cross-module test definitions
    pub test_suites: Vec<IntegrationTestSuite>,

    /// Result collector
    pub results: IntegrationResultCollector,

    /// Resource cleanup tracker
    pub cleanup: CleanupTracker,
}

/// Fixture management for test data
pub struct FixtureManager {
    /// Registered fixtures by name
    fixtures: HashMap<String, Box<dyn TestFixture>>,

    /// Fixture dependencies
    dependencies: HashMap<String, Vec<String>>,

    /// Active fixture instances
    active: Arc<RwLock<HashMap<String, FixtureInstance>>>,

    /// Fixture creation timeout
    creation_timeout: Duration,
}

/// Test fixture trait for reusable test setup
#[async_trait::async_trait]
pub trait TestFixture: Send + Sync {
    /// Fixture identifier
    fn name(&self) -> &str;

    /// Setup the fixture, returning instance data
    async fn setup(&self) -> Result<FixtureData, FixtureError>;

    /// Tear down the fixture
    async fn teardown(&self, data: FixtureData) -> Result<(), FixtureError>;

    /// Dependencies on other fixtures
    fn dependencies(&self) -> Vec<&str> {
        vec![]
    }
}

/// Module orchestrator for integration scenarios
pub struct ModuleOrchestrator {
    /// Available module instances
    modules: HashMap<ModuleId, Arc<dyn ModuleInstance>>,

    /// Inter-module connections
    connections: Vec<ModuleConnection>,

    /// Health check registry
    health_checks: HashMap<ModuleId, Box<dyn HealthCheck>>,

    /// Startup order (topologically sorted)
    startup_order: Vec<ModuleId>,
}

/// Integration test suite definition
#[derive(Clone, Debug)]
pub struct IntegrationTestSuite {
    /// Suite identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Required modules
    pub required_modules: Vec<ModuleId>,

    /// Required fixtures
    pub required_fixtures: Vec<String>,

    /// Test cases in this suite
    pub test_cases: Vec<IntegrationTestCase>,

    /// Suite-level timeout
    pub timeout: Duration,

    /// Parallel execution allowed
    pub parallel: bool,
}

/// Individual integration test case
#[derive(Clone, Debug)]
pub struct IntegrationTestCase {
    /// Test identifier
    pub id: String,

    /// Test description
    pub description: String,

    /// Modules under test
    pub modules: Vec<ModuleId>,

    /// Setup actions
    pub setup: Vec<TestAction>,

    /// Test actions
    pub actions: Vec<TestAction>,

    /// Assertions
    pub assertions: Vec<TestAssertion>,

    /// Cleanup actions
    pub cleanup: Vec<TestAction>,

    /// Expected duration
    pub expected_duration: Duration,
}

/// Isolation manager ensures test independence
pub struct IsolationManager {
    /// Database isolation strategy
    database_strategy: DatabaseIsolation,

    /// File system isolation
    filesystem_strategy: FilesystemIsolation,

    /// Network isolation
    network_strategy: NetworkIsolation,

    /// Memory isolation
    memory_strategy: MemoryIsolation,
}

#[derive(Clone, Debug)]
pub enum DatabaseIsolation {
    /// Each test uses separate database
    SeparateDatabase,
    /// Transaction rollback after each test
    TransactionRollback,
    /// Snapshot and restore
    SnapshotRestore,
}
```

#### 3.2.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-002-01 | Integration tests validate all module interface contracts | Contract test presence per module pair |
| AC-TEST-002-02 | Fixtures are properly cleaned up after each test | Resource leak detection |
| AC-TEST-002-03 | Cross-module coverage exceeds 80% | Coverage report for integration paths |
| AC-TEST-002-04 | Tests are isolated and can run in parallel | Parallel execution without failures |
| AC-TEST-002-05 | Full integration suite completes in <15 minutes | CI timing measurement |
| AC-TEST-002-06 | Failed tests provide clear module interaction traces | Failure output inspection |

#### 3.2.3 Verification Method

```bash
# Run integration tests
cargo nextest run --workspace --test '*integration*' \
    --retries 0 \
    --fail-fast \
    --status-level all

# Verify isolation
cargo nextest run --workspace --test '*integration*' \
    --jobs $(nproc) \
    --no-capture
# Expected: All tests pass with parallel execution
```

---

### REQ-TEST-003: Performance Benchmark Suite

**Priority**: High
**Category**: Performance Validation
**Description**: The system SHALL provide a comprehensive performance benchmarking suite using criterion that measures latency, throughput, and resource utilization for all critical paths.

#### 3.3.1 PerformanceBenchmarks Structure

```rust
use std::collections::HashMap;
use std::time::Duration;
use criterion::{Criterion, BenchmarkGroup, BatchSize, Throughput};

/// Performance benchmarking suite using criterion.
///
/// Measures and tracks performance characteristics of critical
/// system operations with statistical rigor.
///
/// Constraint: Benchmark variance < 2%
/// Constraint: Regression detection with 95% confidence
pub struct PerformanceBenchmarks {
    /// Benchmark registry
    pub benchmarks: HashMap<String, BenchmarkDefinition>,

    /// Baseline measurements for regression detection
    pub baselines: BaselineRegistry,

    /// Statistical configuration
    pub stats_config: StatisticalConfig,

    /// Resource monitoring during benchmarks
    pub resource_monitor: ResourceMonitor,

    /// Benchmark result storage
    pub results: BenchmarkResultStore,

    /// Regression detector
    pub regression_detector: RegressionDetector,
}

/// Benchmark definition
#[derive(Clone, Debug)]
pub struct BenchmarkDefinition {
    /// Benchmark identifier
    pub id: String,

    /// Category (latency, throughput, memory)
    pub category: BenchmarkCategory,

    /// Target module
    pub module: ModuleId,

    /// Operation being benchmarked
    pub operation: String,

    /// Expected latency (for regression)
    pub expected_latency: Option<Duration>,

    /// Expected throughput (ops/sec)
    pub expected_throughput: Option<f64>,

    /// Warmup iterations
    pub warmup_iterations: usize,

    /// Measurement iterations
    pub measurement_iterations: usize,

    /// Sample size per iteration
    pub sample_size: usize,
}

#[derive(Clone, Debug)]
pub enum BenchmarkCategory {
    Latency,
    Throughput,
    Memory,
    CPUCycles,
    CacheHits,
    IOOperations,
}

/// Statistical configuration for benchmarks
#[derive(Clone, Debug)]
pub struct StatisticalConfig {
    /// Confidence level for statistical tests
    pub confidence_level: f64,  // Default: 0.95

    /// Noise threshold (ignore changes below this)
    pub noise_threshold: f64,  // Default: 0.02 (2%)

    /// Outlier detection sensitivity
    pub outlier_sensitivity: f64,  // Default: 3.0 (3 sigma)

    /// Minimum samples for significance
    pub min_samples: usize,  // Default: 100

    /// Maximum relative error allowed
    pub max_relative_error: f64,  // Default: 0.02 (2%)
}

/// Baseline registry for regression detection
pub struct BaselineRegistry {
    /// Stored baselines by benchmark ID
    baselines: HashMap<String, BaselineMeasurement>,

    /// Baseline storage path
    storage_path: std::path::PathBuf,

    /// Baseline version
    version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaselineMeasurement {
    /// Mean latency in nanoseconds
    pub mean_ns: f64,

    /// Standard deviation
    pub std_dev_ns: f64,

    /// Median latency
    pub median_ns: f64,

    /// P99 latency
    pub p99_ns: f64,

    /// Throughput (ops/sec)
    pub throughput: f64,

    /// Timestamp of measurement
    pub timestamp: i64,

    /// Git commit hash
    pub commit: String,
}

/// Regression detection
pub struct RegressionDetector {
    /// Statistical test to use
    test_type: StatisticalTest,

    /// Significance level (p-value threshold)
    significance_level: f64,

    /// Minimum effect size for regression
    min_effect_size: f64,
}

#[derive(Clone, Debug)]
pub enum StatisticalTest {
    TTest,
    MannWhitneyU,
    KolmogorovSmirnov,
}

/// Critical path benchmarks to implement
pub const CRITICAL_BENCHMARKS: &[&str] = &[
    // Core Operations
    "core::memory_store_single",
    "core::memory_store_batch_1000",
    "core::memory_recall_by_id",
    "core::memory_recall_by_similarity",

    // Embedding Operations
    "embedding::generate_single",
    "embedding::generate_batch_100",
    "embedding::similarity_search_1k",
    "embedding::similarity_search_100k",

    // Graph Operations
    "graph::node_insert",
    "graph::edge_insert",
    "graph::traverse_depth_3",
    "graph::traverse_depth_5",
    "graph::shortest_path",

    // MCP Operations
    "mcp::inject_context_simple",
    "mcp::inject_context_complex",
    "mcp::store_memory",
    "mcp::recall_memory",

    // GPU Operations (when available)
    "cuda::embedding_batch_gpu",
    "cuda::similarity_matrix_gpu",

    // I/O Operations
    "storage::rocksdb_write",
    "storage::rocksdb_read",
    "storage::batch_write_1000",
];
```

#### 3.3.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-003-01 | All critical paths have defined benchmarks | Benchmark coverage check |
| AC-TEST-003-02 | Benchmark variance is <2% across runs | Statistical analysis |
| AC-TEST-003-03 | Regressions are detected with 95% confidence | Regression test with known slowdown |
| AC-TEST-003-04 | Baselines are stored and versioned | Baseline file verification |
| AC-TEST-003-05 | CPU and memory metrics are captured | Resource monitor output |
| AC-TEST-003-06 | Reports include percentile distributions | Report format verification |

#### 3.3.3 Verification Method

```bash
# Run benchmarks
cargo bench --workspace --bench '*' -- \
    --sample-size 100 \
    --measurement-time 5 \
    --warm-up-time 1

# Compare against baseline
cargo bench --workspace --bench '*' -- \
    --save-baseline main \
    --baseline main

# Verify variance
# Check criterion output for "change within noise threshold"
```

---

### REQ-TEST-004: Load Testing Framework

**Priority**: High
**Category**: Capacity Validation
**Description**: The system SHALL provide a load testing framework that validates system behavior under concurrent load, identifies breaking points, and measures degradation patterns.

#### 3.4.1 LoadTester Structure

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, RwLock};

/// Load testing framework for capacity validation.
///
/// Simulates concurrent client sessions, measures system behavior
/// under load, and identifies capacity limits.
///
/// Constraint: Support 10,000 concurrent sessions
/// Constraint: Accurate latency percentiles under load
pub struct LoadTester {
    /// Load profile configuration
    pub profile: LoadProfile,

    /// Client simulator pool
    pub client_pool: ClientPool,

    /// Real-time metrics collector
    pub metrics: LoadMetricsCollector,

    /// Saturation detector
    pub saturation_detector: SaturationDetector,

    /// Report generator
    pub reporter: LoadTestReporter,

    /// Test scenarios
    pub scenarios: Vec<LoadScenario>,
}

/// Load profile configuration
#[derive(Clone, Debug)]
pub struct LoadProfile {
    /// Profile type
    pub profile_type: LoadProfileType,

    /// Target requests per second
    pub target_rps: u64,

    /// Maximum concurrent clients
    pub max_concurrent: usize,

    /// Ramp-up duration
    pub ramp_up: Duration,

    /// Sustained load duration
    pub sustained: Duration,

    /// Ramp-down duration
    pub ramp_down: Duration,

    /// Think time between requests
    pub think_time: ThinkTime,
}

#[derive(Clone, Debug)]
pub enum LoadProfileType {
    /// Constant load
    Constant,
    /// Linear ramp
    Ramp,
    /// Step function
    Step { step_size: usize, step_duration: Duration },
    /// Spike pattern
    Spike { spike_multiplier: f64, spike_duration: Duration },
    /// Stress test (find breaking point)
    Stress { increment: usize, threshold_errors: f64 },
}

#[derive(Clone, Debug)]
pub enum ThinkTime {
    Fixed(Duration),
    Uniform { min: Duration, max: Duration },
    Exponential { mean: Duration },
    None,
}

/// Simulated client pool
pub struct ClientPool {
    /// Pool capacity
    capacity: usize,

    /// Active clients
    active: Arc<Semaphore>,

    /// Client factory
    factory: Box<dyn ClientFactory>,

    /// Connection reuse
    connection_reuse: bool,
}

/// Load scenario definition
#[derive(Clone, Debug)]
pub struct LoadScenario {
    /// Scenario identifier
    pub id: String,

    /// Scenario name
    pub name: String,

    /// Weight (probability of selection)
    pub weight: f64,

    /// Request sequence
    pub requests: Vec<LoadRequest>,

    /// Success criteria
    pub success_criteria: SuccessCriteria,
}

/// Individual load request
#[derive(Clone, Debug)]
pub struct LoadRequest {
    /// Request type
    pub request_type: RequestType,

    /// MCP tool name
    pub tool: String,

    /// Request parameters (template with variables)
    pub params: serde_json::Value,

    /// Expected response time
    pub expected_latency: Duration,

    /// Timeout
    pub timeout: Duration,

    /// Validation rules
    pub validation: Vec<ResponseValidation>,
}

#[derive(Clone, Debug)]
pub enum RequestType {
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Mixed read/write
    ReadWrite,
    /// Heavy computation
    Compute,
}

/// Success criteria for load tests
#[derive(Clone, Debug)]
pub struct SuccessCriteria {
    /// Maximum error rate
    pub max_error_rate: f64,  // Default: 0.01 (1%)

    /// Maximum P99 latency
    pub max_p99_latency: Duration,

    /// Minimum throughput
    pub min_throughput: f64,

    /// Maximum saturation (queue depth)
    pub max_saturation: f64,
}

/// Saturation detection
pub struct SaturationDetector {
    /// Queue depth monitor
    queue_monitor: QueueDepthMonitor,

    /// Latency trend analyzer
    latency_analyzer: LatencyTrendAnalyzer,

    /// Error rate monitor
    error_monitor: ErrorRateMonitor,

    /// Saturation threshold
    saturation_threshold: f64,
}

/// Real-time metrics during load test
pub struct LoadMetricsCollector {
    /// Requests per second
    pub rps: AtomicU64,

    /// Active connections
    pub active_connections: AtomicU64,

    /// Latency histogram (HDR Histogram)
    pub latency_histogram: Arc<RwLock<hdrhistogram::Histogram<u64>>>,

    /// Error count by type
    pub errors: DashMap<String, AtomicU64>,

    /// Throughput bytes/sec
    pub throughput_bytes: AtomicU64,

    /// Collection interval
    pub interval: Duration,

    /// Time series data points
    pub time_series: Arc<RwLock<Vec<MetricsSnapshot>>>,
}
```

#### 3.4.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-004-01 | Load tester supports 10,000 concurrent sessions | Concurrent connection count |
| AC-TEST-004-02 | Latency percentiles are accurate (P50, P95, P99, P99.9) | HDR Histogram verification |
| AC-TEST-004-03 | Saturation point is accurately detected | Saturation threshold crossing |
| AC-TEST-004-04 | Multiple load profiles are supported | Profile execution verification |
| AC-TEST-004-05 | Real-time metrics are available during test | Metrics endpoint check |
| AC-TEST-004-06 | Breaking point is identified in stress tests | Stress test output analysis |

#### 3.4.3 Verification Method

```bash
# Run load test with specific profile
cargo run --release --bin load-tester -- \
    --profile stress \
    --max-concurrent 10000 \
    --duration 300s \
    --scenario mixed_workload

# Verify capacity
# Output should show:
# - Maximum sustainable RPS
# - Saturation point
# - P99 latency at saturation
```

---

### REQ-TEST-005: Chaos Engineering Framework

**Priority**: High
**Category**: Resilience Validation
**Description**: The system SHALL provide a chaos engineering framework that injects faults, validates recovery mechanisms, and ensures system resilience under failure conditions.

#### 3.5.1 ChaosEngine Structure

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

/// Chaos engineering framework for resilience testing.
///
/// Injects controlled failures, monitors system behavior,
/// and validates recovery mechanisms.
///
/// Constraint: Mean time to recovery < 30 seconds
/// Constraint: No data corruption under chaos
pub struct ChaosEngine {
    /// Fault injection registry
    pub fault_registry: FaultRegistry,

    /// Chaos experiment definitions
    pub experiments: Vec<ChaosExperiment>,

    /// System monitor during chaos
    pub monitor: ChaosMonitor,

    /// Recovery validator
    pub recovery_validator: RecoveryValidator,

    /// Safety controls
    pub safety: SafetyControls,

    /// Experiment scheduler
    pub scheduler: ChaosScheduler,
}

/// Fault injection types
#[derive(Clone, Debug)]
pub enum FaultType {
    /// Process crash
    ProcessCrash {
        target: ProcessTarget,
        restart_delay: Option<Duration>,
    },

    /// Network partition
    NetworkPartition {
        partitions: Vec<Vec<String>>,
        duration: Duration,
    },

    /// Network latency injection
    NetworkLatency {
        target: NetworkTarget,
        latency: Duration,
        jitter: Duration,
    },

    /// Packet loss
    PacketLoss {
        target: NetworkTarget,
        loss_rate: f64,
    },

    /// Disk failure
    DiskFailure {
        target: String,
        failure_type: DiskFailureType,
    },

    /// CPU stress
    CPUStress {
        utilization: f64,
        duration: Duration,
    },

    /// Memory pressure
    MemoryPressure {
        allocation_mb: usize,
        duration: Duration,
    },

    /// Clock skew
    ClockSkew {
        offset: Duration,
        direction: ClockDirection,
    },

    /// Dependency failure
    DependencyFailure {
        service: String,
        failure_mode: DependencyFailureMode,
    },

    /// Corruption injection
    DataCorruption {
        target: DataTarget,
        corruption_type: CorruptionType,
    },
}

#[derive(Clone, Debug)]
pub enum DiskFailureType {
    ReadOnly,
    Full,
    Slow { latency: Duration },
    Corrupt { rate: f64 },
}

#[derive(Clone, Debug)]
pub enum DependencyFailureMode {
    Timeout,
    Error { code: i32 },
    Slow { latency: Duration },
    Partial { success_rate: f64 },
}

/// Chaos experiment definition
#[derive(Clone, Debug)]
pub struct ChaosExperiment {
    /// Experiment identifier
    pub id: String,

    /// Experiment name
    pub name: String,

    /// Hypothesis (what we expect to happen)
    pub hypothesis: String,

    /// Faults to inject
    pub faults: Vec<FaultInjection>,

    /// Duration of chaos
    pub duration: Duration,

    /// Steady state definition
    pub steady_state: SteadyStateDefinition,

    /// Recovery expectations
    pub recovery: RecoveryExpectation,

    /// Abort conditions
    pub abort_conditions: Vec<AbortCondition>,
}

/// Steady state definition for validation
#[derive(Clone, Debug)]
pub struct SteadyStateDefinition {
    /// Metrics that define normal operation
    pub metrics: Vec<SteadyStateMetric>,

    /// Validation interval
    pub check_interval: Duration,

    /// Required duration at steady state
    pub required_duration: Duration,
}

#[derive(Clone, Debug)]
pub struct SteadyStateMetric {
    /// Metric name
    pub name: String,

    /// Expected range
    pub min: f64,
    pub max: f64,

    /// Tolerance for violations
    pub tolerance: f64,
}

/// Recovery validation
pub struct RecoveryValidator {
    /// Expected recovery time
    expected_recovery: Duration,

    /// Data integrity checker
    integrity_checker: DataIntegrityChecker,

    /// Service health checker
    health_checker: ServiceHealthChecker,

    /// Recovery metrics
    recovery_metrics: RecoveryMetrics,
}

/// Safety controls for chaos experiments
pub struct SafetyControls {
    /// Emergency stop channel
    emergency_stop: broadcast::Sender<()>,

    /// Production environment detection
    environment_guard: EnvironmentGuard,

    /// Blast radius limits
    blast_radius: BlastRadiusConfig,

    /// Automatic rollback
    auto_rollback: bool,

    /// Maximum experiment duration
    max_duration: Duration,
}

/// Blast radius configuration
#[derive(Clone, Debug)]
pub struct BlastRadiusConfig {
    /// Maximum percentage of instances affected
    pub max_affected_pct: f64,

    /// Maximum number of instances affected
    pub max_affected_count: usize,

    /// Protected instances (never affected)
    pub protected: Vec<String>,

    /// Required healthy instances
    pub min_healthy: usize,
}
```

#### 3.5.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-005-01 | System recovers from process crash within 30 seconds | Recovery timing measurement |
| AC-TEST-005-02 | No data corruption under any chaos scenario | Data integrity validation |
| AC-TEST-005-03 | Network partition handling maintains consistency | Consistency check post-partition |
| AC-TEST-005-04 | Safety controls prevent runaway chaos | Abort condition testing |
| AC-TEST-005-05 | Steady state is restored after fault removal | Steady state metrics validation |
| AC-TEST-005-06 | Experiments are reproducible | Same experiment, same results |

#### 3.5.3 Verification Method

```bash
# Run chaos experiment
cargo run --release --bin chaos-engine -- \
    --experiment process_crash_recovery \
    --safety-mode enabled \
    --max-duration 300s

# Verify recovery
# Output should show:
# - Time to detection
# - Time to recovery
# - Data integrity status
# - Steady state restoration
```

---

### REQ-TEST-006: Production Configuration

**Priority**: Critical
**Category**: Deployment Infrastructure
**Description**: The system SHALL provide production-ready deployment configurations including Docker images, Kubernetes manifests, and infrastructure-as-code templates.

#### 3.6.1 ProductionConfig Structure

```rust
use std::collections::HashMap;
use std::path::PathBuf;

/// Production deployment configuration.
///
/// Provides container images, orchestration configs,
/// and infrastructure templates for production deployment.
pub struct ProductionConfig {
    /// Docker configuration
    pub docker: DockerConfig,

    /// Kubernetes configuration
    pub kubernetes: KubernetesConfig,

    /// Environment configurations
    pub environments: HashMap<String, EnvironmentConfig>,

    /// Resource requirements
    pub resources: ResourceRequirements,

    /// Health check configuration
    pub health: HealthCheckConfig,

    /// Scaling configuration
    pub scaling: ScalingConfig,
}

/// Docker configuration
#[derive(Clone, Debug)]
pub struct DockerConfig {
    /// Base image
    pub base_image: String,  // Default: "rust:1.75-slim-bookworm"

    /// Multi-stage build layers
    pub build_stages: Vec<BuildStage>,

    /// Runtime image
    pub runtime_image: String,  // Default: "debian:bookworm-slim"

    /// Exposed ports
    pub ports: Vec<PortMapping>,

    /// Volume mounts
    pub volumes: Vec<VolumeMount>,

    /// Environment variables
    pub env: HashMap<String, String>,

    /// Health check command
    pub healthcheck: DockerHealthCheck,

    /// Image labels
    pub labels: HashMap<String, String>,

    /// Security options
    pub security: DockerSecurity,
}

/// Docker multi-stage build
#[derive(Clone, Debug)]
pub struct BuildStage {
    pub name: String,
    pub from: String,
    pub commands: Vec<String>,
    pub copy_from: Option<String>,
}

/// Kubernetes configuration
#[derive(Clone, Debug)]
pub struct KubernetesConfig {
    /// Namespace
    pub namespace: String,

    /// Deployment configuration
    pub deployment: DeploymentConfig,

    /// Service configuration
    pub service: ServiceConfig,

    /// ConfigMap definitions
    pub config_maps: Vec<ConfigMapDef>,

    /// Secret references
    pub secrets: Vec<SecretRef>,

    /// Ingress configuration
    pub ingress: Option<IngressConfig>,

    /// Horizontal Pod Autoscaler
    pub hpa: Option<HPAConfig>,

    /// Pod Disruption Budget
    pub pdb: Option<PDBConfig>,

    /// Network policies
    pub network_policies: Vec<NetworkPolicy>,
}

/// Deployment configuration
#[derive(Clone, Debug)]
pub struct DeploymentConfig {
    /// Replica count
    pub replicas: u32,

    /// Update strategy
    pub strategy: UpdateStrategy,

    /// Pod template
    pub pod_template: PodTemplate,

    /// Affinity rules
    pub affinity: Option<AffinityRules>,

    /// Tolerations
    pub tolerations: Vec<Toleration>,
}

#[derive(Clone, Debug)]
pub enum UpdateStrategy {
    RollingUpdate {
        max_unavailable: String,
        max_surge: String,
    },
    Recreate,
}

/// Pod template specification
#[derive(Clone, Debug)]
pub struct PodTemplate {
    /// Container specifications
    pub containers: Vec<ContainerSpec>,

    /// Init containers
    pub init_containers: Vec<ContainerSpec>,

    /// Volumes
    pub volumes: Vec<VolumeSpec>,

    /// Service account
    pub service_account: String,

    /// Security context
    pub security_context: PodSecurityContext,

    /// DNS policy
    pub dns_policy: String,

    /// Termination grace period
    pub termination_grace_period: u32,
}

/// Container specification
#[derive(Clone, Debug)]
pub struct ContainerSpec {
    pub name: String,
    pub image: String,
    pub image_pull_policy: String,
    pub ports: Vec<ContainerPort>,
    pub env: Vec<EnvVar>,
    pub env_from: Vec<EnvFromSource>,
    pub resources: ResourceLimits,
    pub liveness_probe: Option<Probe>,
    pub readiness_probe: Option<Probe>,
    pub startup_probe: Option<Probe>,
    pub volume_mounts: Vec<VolumeMount>,
    pub security_context: ContainerSecurityContext,
}

/// Resource limits
#[derive(Clone, Debug)]
pub struct ResourceLimits {
    /// CPU request
    pub cpu_request: String,  // e.g., "500m"

    /// CPU limit
    pub cpu_limit: String,  // e.g., "2"

    /// Memory request
    pub memory_request: String,  // e.g., "1Gi"

    /// Memory limit
    pub memory_limit: String,  // e.g., "4Gi"

    /// GPU request (optional)
    pub gpu_request: Option<String>,
}

/// Health check probes
#[derive(Clone, Debug)]
pub struct Probe {
    pub probe_type: ProbeType,
    pub initial_delay_seconds: u32,
    pub period_seconds: u32,
    pub timeout_seconds: u32,
    pub success_threshold: u32,
    pub failure_threshold: u32,
}

#[derive(Clone, Debug)]
pub enum ProbeType {
    HttpGet { path: String, port: u32 },
    TcpSocket { port: u32 },
    Exec { command: Vec<String> },
    Grpc { port: u32 },
}

/// Horizontal Pod Autoscaler
#[derive(Clone, Debug)]
pub struct HPAConfig {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub metrics: Vec<ScalingMetric>,
    pub behavior: ScalingBehavior,
}

#[derive(Clone, Debug)]
pub enum ScalingMetric {
    CPU { target_utilization: u32 },
    Memory { target_utilization: u32 },
    Custom { name: String, target_value: String },
}
```

#### 3.6.2 Dockerfile Summary

| Stage | Base Image | Purpose |
|-------|------------|---------|
| builder | rust:1.75-slim-bookworm | Compile with dependencies |
| runtime | debian:bookworm-slim | Minimal production image |

**Key Features**:
- Multi-stage build for minimal image size
- Non-root user execution
- Health check on port 3000
- Production config mount

#### 3.6.3 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-006-01 | Docker image builds successfully | Docker build verification |
| AC-TEST-006-02 | Container starts in <10 seconds | Startup timing measurement |
| AC-TEST-006-03 | Health checks pass within 30 seconds | Health probe verification |
| AC-TEST-006-04 | Kubernetes manifests are valid | kubectl dry-run |
| AC-TEST-006-05 | Resource limits are appropriate | Performance testing |
| AC-TEST-006-06 | Security contexts are properly configured | Security scan |

---

### REQ-TEST-007: Monitoring Infrastructure

**Priority**: High
**Category**: Observability
**Description**: The system SHALL expose Prometheus metrics, structured logs, and distributed tracing for production observability.

#### 3.7.1 MonitoringConfig Structure

```rust
use std::collections::HashMap;
use std::time::Duration;

/// Monitoring and observability configuration.
///
/// Provides metrics exposition, structured logging,
/// and distributed tracing for production systems.
pub struct MonitoringConfig {
    /// Prometheus metrics configuration
    pub prometheus: PrometheusConfig,

    /// Structured logging configuration
    pub logging: LoggingConfig,

    /// Distributed tracing configuration
    pub tracing: TracingConfig,

    /// Alert rule definitions
    pub alerts: Vec<AlertRule>,

    /// Dashboard definitions
    pub dashboards: Vec<Dashboard>,
}

/// Prometheus metrics configuration
#[derive(Clone, Debug)]
pub struct PrometheusConfig {
    /// Metrics endpoint path
    pub endpoint: String,  // Default: "/metrics"

    /// Metrics port
    pub port: u16,  // Default: 9090

    /// Metric prefix
    pub prefix: String,  // Default: "context_graph_"

    /// Default labels
    pub default_labels: HashMap<String, String>,

    /// Histogram buckets for latencies
    pub latency_buckets: Vec<f64>,

    /// Metric definitions
    pub metrics: Vec<MetricDefinition>,
}

/// Metric definition
#[derive(Clone, Debug)]
pub struct MetricDefinition {
    /// Metric name
    pub name: String,

    /// Metric type
    pub metric_type: MetricType,

    /// Help text
    pub help: String,

    /// Labels
    pub labels: Vec<String>,

    /// Buckets (for histograms)
    pub buckets: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Required metrics categories: Request (3), Memory (3), Embedding (3),
/// Graph (3), Resource (3), Health (2) = 17 total Prometheus metrics

/// Alert rule definition
#[derive(Clone, Debug)]
pub struct AlertRule {
    /// Alert name
    pub name: String,

    /// Alert expression (PromQL)
    pub expr: String,

    /// Duration before firing
    pub for_duration: Duration,

    /// Severity level
    pub severity: AlertSeverity,

    /// Alert labels
    pub labels: HashMap<String, String>,

    /// Alert annotations
    pub annotations: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Page,
}

/// Required alerts: ContextGraphDown, HighErrorRate, VeryHighErrorRate,
/// HighLatency, VeryHighLatency, HighMemoryUsage, HighCPUUsage,
/// HighQueueDepth, NearCapacity (9 total with thresholds per SLO)
```

#### 3.7.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-007-01 | All required metrics are exposed | Metrics endpoint verification |
| AC-TEST-007-02 | Metrics have correct types and labels | Prometheus scrape test |
| AC-TEST-007-03 | Alert rules are syntactically valid | promtool check rules |
| AC-TEST-007-04 | Structured logs are in JSON format | Log output verification |
| AC-TEST-007-05 | Trace spans are properly correlated | Trace ID propagation test |
| AC-TEST-007-06 | Dashboards render without errors | Grafana dashboard validation |

---

### REQ-TEST-008: Security Testing

**Priority**: Critical
**Category**: Security Validation
**Description**: The system SHALL undergo security testing including dependency auditing, fuzzing, and penetration testing to identify vulnerabilities.

#### 3.8.1 SecurityTestConfig Structure

```rust
use std::collections::HashMap;
use std::path::PathBuf;

/// Security testing configuration.
///
/// Provides dependency auditing, fuzzing, and security
/// scanning for vulnerability identification.
pub struct SecurityTestConfig {
    /// Dependency audit configuration
    pub dependency_audit: DependencyAuditConfig,

    /// Fuzzing configuration
    pub fuzzing: FuzzingConfig,

    /// Static analysis configuration
    pub static_analysis: StaticAnalysisConfig,

    /// Penetration testing configuration
    pub penetration: PenetrationTestConfig,
}

/// Dependency audit configuration
#[derive(Clone, Debug)]
pub struct DependencyAuditConfig {
    /// Audit tool (cargo-audit)
    pub tool: String,

    /// Advisory database source
    pub advisory_db: String,

    /// Severity threshold for failure
    pub fail_threshold: SecuritySeverity,

    /// Ignored advisories
    pub ignore: Vec<String>,

    /// Report output path
    pub report_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecuritySeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Fuzzing configuration
#[derive(Clone, Debug)]
pub struct FuzzingConfig {
    /// Fuzzing tool (cargo-fuzz, afl)
    pub tool: FuzzingTool,

    /// Fuzz targets
    pub targets: Vec<FuzzTarget>,

    /// Maximum duration per target
    pub max_duration: std::time::Duration,

    /// Corpus directory
    pub corpus_dir: PathBuf,

    /// Artifact directory
    pub artifact_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub enum FuzzingTool {
    CargoFuzz,
    AFL,
    LibFuzzer,
}

/// Fuzz target definition
#[derive(Clone, Debug)]
pub struct FuzzTarget {
    /// Target name
    pub name: String,

    /// Target function/module
    pub function: String,

    /// Input type
    pub input_type: FuzzInputType,

    /// Maximum input size
    pub max_size: usize,

    /// Seed corpus
    pub seeds: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub enum FuzzInputType {
    Bytes,
    Json,
    Protobuf,
    Custom(String),
}

/// Required fuzz targets
pub const REQUIRED_FUZZ_TARGETS: &[&str] = &[
    "fuzz_json_rpc_parse",
    "fuzz_mcp_request_validate",
    "fuzz_memory_node_deserialize",
    "fuzz_embedding_input",
    "fuzz_graph_query_parse",
    "fuzz_utl_decompress",
];
```

#### 3.8.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-008-01 | No high/critical dependency vulnerabilities | cargo audit output |
| AC-TEST-008-02 | All fuzz targets pass 1 hour of fuzzing | Fuzzing run logs |
| AC-TEST-008-03 | Static analysis has zero critical findings | clippy/semgrep output |
| AC-TEST-008-04 | Input validation blocks all injection attempts | Injection test suite |
| AC-TEST-008-05 | Rate limiting prevents DoS | DoS simulation test |
| AC-TEST-008-06 | Authentication cannot be bypassed | Auth bypass test suite |

---

### REQ-TEST-009: Documentation Generation

**Priority**: Medium
**Category**: Documentation
**Description**: The system SHALL automatically generate API documentation, architecture diagrams, and operational runbooks from code and configuration.

#### 3.9.1 DocumentationConfig Structure

```rust
use std::path::PathBuf;

/// Documentation generation configuration.
///
/// Automatically generates API docs, architecture diagrams,
/// and operational documentation.
pub struct DocumentationConfig {
    /// API documentation (rustdoc)
    pub api_docs: ApiDocsConfig,

    /// Architecture diagrams
    pub diagrams: DiagramConfig,

    /// Operational runbooks
    pub runbooks: RunbookConfig,

    /// Output directory
    pub output_dir: PathBuf,
}

/// API documentation configuration
#[derive(Clone, Debug)]
pub struct ApiDocsConfig {
    /// Enable private items
    pub document_private: bool,

    /// Enable hidden items
    pub document_hidden: bool,

    /// Default theme
    pub theme: String,

    /// Additional markdown files
    pub additional_pages: Vec<PathBuf>,

    /// Example code extraction
    pub extract_examples: bool,
}

/// Diagram generation configuration
#[derive(Clone, Debug)]
pub struct DiagramConfig {
    /// Diagram format
    pub format: DiagramFormat,

    /// Module dependency diagram
    pub dependency_diagram: bool,

    /// Sequence diagrams for key flows
    pub sequence_diagrams: Vec<String>,

    /// Architecture overview
    pub architecture_overview: bool,
}

#[derive(Clone, Debug)]
pub enum DiagramFormat {
    Mermaid,
    PlantUML,
    Graphviz,
}

/// Runbook configuration
#[derive(Clone, Debug)]
pub struct RunbookConfig {
    /// Runbook templates
    pub templates: Vec<RunbookTemplate>,

    /// Alert-to-runbook mapping
    pub alert_mapping: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct RunbookTemplate {
    pub name: String,
    pub description: String,
    pub steps: Vec<RunbookStep>,
    pub escalation: Option<EscalationPath>,
}
```

#### 3.9.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-009-01 | API documentation is complete | rustdoc coverage check |
| AC-TEST-009-02 | All public types are documented | Documentation lint |
| AC-TEST-009-03 | Architecture diagrams are current | Diagram/code comparison |
| AC-TEST-009-04 | Runbooks exist for all alerts | Alert-runbook mapping |
| AC-TEST-009-05 | Examples compile and run | Example test execution |
| AC-TEST-009-06 | Documentation builds without warnings | Build output verification |

---

### REQ-TEST-010: Continuous Integration Pipeline

**Priority**: Critical
**Category**: Automation
**Description**: The system SHALL provide a comprehensive CI/CD pipeline that automates testing, benchmarking, security scanning, and deployment.

#### 3.10.1 CI/CD Pipeline Stages

| Stage | Name | Duration | Dependencies | Actions |
|-------|------|----------|--------------|---------|
| 1 | quick-checks | <2 min | None | fmt, clippy, check |
| 2 | unit-tests | <5 min | Stage 1 | cargo nextest --lib |
| 3 | coverage | <10 min | Stage 1 | llvm-cov, fail-under 90% |
| 4 | integration-tests | <15 min | Stage 2 | nextest integration |
| 5 | security | <3 min | Stage 1 | cargo-audit |
| 6 | benchmarks | Weekly | Stage 4 | criterion |
| 7 | build-release | <5 min | Stage 4+5 | cargo build --release |
| 8 | docker | <10 min | Stage 7 | build + trivy scan |

**Total Pipeline Duration**: <30 minutes (parallel stages)

#### 3.10.2 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-010-01 | CI runs on every PR | GitHub Actions history |
| AC-TEST-010-02 | All stages complete in <30 minutes | Pipeline timing |
| AC-TEST-010-03 | Coverage threshold is enforced | Coverage gate check |
| AC-TEST-010-04 | Security scan blocks vulnerabilities | Security gate check |
| AC-TEST-010-05 | Benchmarks detect regressions | Benchmark comparison |
| AC-TEST-010-06 | Docker images are scanned | Trivy output |

---

### REQ-TEST-011: Marblestone Feature Tests

**Priority**: High
**Category**: Feature Validation
**Description**: The system SHALL validate all Marblestone-inspired cognitive architecture features including neurotransmitter-weighted edges, lifecycle lambda weights, formal verification, amortized shortcuts, steering dopamine feedback, omnidirectional inference, and new MCP tools.

#### 3.11.1 Marblestone Feature Test Requirements

### Marblestone Feature Tests

#### T-MARB-001: Neurotransmitter Edge Weights
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-001-01 | Create edge with NeurotransmitterWeights | Edge stores excitatory/inhibitory/modulatory values |
| T-MARB-001-02 | get_modulated_weight() with high excitatory | Returns weight > base weight |
| T-MARB-001-03 | get_modulated_weight() with high inhibitory | Returns weight < base weight |
| T-MARB-001-04 | domain_aware_search() by domain | Results modulated by domain profile |

**Cross-Reference**: Module 2 (Core Infrastructure) - NeurotransmitterWeights struct, Module 4 (Knowledge Graph) - Edge weight modulation

#### T-MARB-002: Lifecycle Lambda Weights
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-002-01 | Infancy stage lambda weights | lambda_DS=0.7, lambda_DC=0.3 |
| T-MARB-002-02 | Growth stage lambda weights | lambda_DS=0.5, lambda_DC=0.5 |
| T-MARB-002-03 | Maturity stage lambda weights | lambda_DS=0.3, lambda_DC=0.7 |
| T-MARB-002-04 | Stage transition updates lambdas | Weights change on stage change |

**Cross-Reference**: Module 6 (Bio-Nervous System) - Lifecycle stages, Module 10 (Neuromodulation) - Lambda weight adaptation

#### T-MARB-003: Formal Verification Layer
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-003-01 | Verify valid code node | Returns Verified |
| T-MARB-003-02 | Verify code with bounds error | Returns Failed with counterexample |
| T-MARB-003-03 | Verify non-code content | Returns NotApplicable |
| T-MARB-003-04 | Verification timeout | Returns Timeout after configured ms |

**Cross-Reference**: Module 11 (Immune System) - Code verification integration, Module 13 (MCP Hardening) - Verification security boundaries

#### T-MARB-004: Amortized Shortcut Creation
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-004-01 | Create shortcut from 3-hop path | Shortcut edge created with is_amortized_shortcut=true |
| T-MARB-004-02 | Reject 2-hop path | No shortcut created |
| T-MARB-004-03 | Reject low confidence path | No shortcut for confidence < 0.7 |
| T-MARB-004-04 | Shortcut weight calculation | Weight = product of path edge weights |

**Cross-Reference**: Module 4 (Knowledge Graph) - Path analysis, Module 9 (Dream Layer) - Shortcut consolidation during memory sleep

#### T-MARB-005: Steering Dopamine Feedback
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-005-01 | Positive steering reward | Dopamine increases by reward * 0.2 |
| T-MARB-005-02 | Negative steering reward | Dopamine decreases by |reward| * 0.1 |
| T-MARB-005-03 | Dopamine clamping | Dopamine stays in [0, 1] |
| T-MARB-005-04 | Steering source tracking | Source (Gardener/Curator/Assessor) logged |

**Cross-Reference**: Module 10 (Neuromodulation) - Dopamine modulator, Module 12 (Active Inference) - Steering integration

#### T-MARB-006: Omnidirectional Inference
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-006-01 | Forward inference A->B | Given A, infers B |
| T-MARB-006-02 | Backward inference B->A | Given B, infers likely A |
| T-MARB-006-03 | Clamped variable inference | Clamped variables unchanged |
| T-MARB-006-04 | Abductive inference | Returns best explanation |

**Cross-Reference**: Module 12 (Active Inference) - OmniInferenceEngine, Module 4 (Knowledge Graph) - Bidirectional traversal

#### T-MARB-007: Steering Subsystem
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-007-01 | SteeringReward returned on store | Every store_memory returns SteeringReward |
| T-MARB-007-02 | Gardener evaluation | Long-term value score computed |
| T-MARB-007-03 | Curator evaluation | Quality score computed |
| T-MARB-007-04 | Thought Assessor evaluation | Immediate relevance computed |

**Cross-Reference**: Module 2 (Core Infrastructure) - SteeringReward type, Module 10 (Neuromodulation) - Evaluation subsystems

#### T-MARB-008: New MCP Tools
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| T-MARB-008-01 | get_steering_feedback MCP call | Returns SteeringFeedbackResponse |
| T-MARB-008-02 | omni_infer MCP call | Returns OmniInferResponse |
| T-MARB-008-03 | verify_code_node MCP call | Returns VerifyCodeResponse |

**Cross-Reference**: Module 13 (MCP Hardening) - Tool security validation, Module 2 (Core Infrastructure) - Response types

#### 3.11.2 Marblestone Integration Test Requirements

| Integration Test ID | Test Scenario | Modules Involved | Expected Behavior |
|---------------------|---------------|------------------|-------------------|
| T-MARB-INT-001 | End-to-end steering feedback flow | 2, 10, 12, 13 | store_memory triggers all evaluators, aggregates SteeringReward, updates dopamine |
| T-MARB-INT-002 | Neurotransmitter-modulated search | 2, 3, 4 | Search results weighted by excitatory/inhibitory balance |
| T-MARB-INT-003 | Lifecycle-aware memory consolidation | 6, 9, 10 | Dream layer respects lifecycle stage lambda weights |
| T-MARB-INT-004 | Formal verification + immune response | 11, 13 | Failed verification triggers immune quarantine |
| T-MARB-INT-005 | Shortcut creation during dreaming | 4, 9 | High-confidence paths become amortized shortcuts |
| T-MARB-INT-006 | Omnidirectional inference with modulation | 4, 10, 12 | Inference respects neuromodulator state |

#### 3.11.3 Marblestone Benchmark Requirements

| Benchmark ID | Operation | Target Latency | Target Throughput |
|--------------|-----------|----------------|-------------------|
| BENCH-MARB-001 | get_modulated_weight() | <1ms | 100K ops/sec |
| BENCH-MARB-002 | domain_aware_search() | <50ms | 1K queries/sec |
| BENCH-MARB-003 | SteeringReward aggregation | <10ms | 10K ops/sec |
| BENCH-MARB-004 | omni_infer (forward) | <100ms | 100 queries/sec |
| BENCH-MARB-005 | omni_infer (backward) | <200ms | 50 queries/sec |
| BENCH-MARB-006 | verify_code_node (small) | <500ms | 10 ops/sec |
| BENCH-MARB-007 | Shortcut creation check | <5ms | 50K ops/sec |

#### 3.11.4 Acceptance Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-TEST-011-01 | All T-MARB unit tests pass | cargo nextest run --test marblestone |
| AC-TEST-011-02 | Integration tests validate cross-module flows | T-MARB-INT test execution |
| AC-TEST-011-03 | Benchmarks meet latency targets | BENCH-MARB criterion output |
| AC-TEST-011-04 | MCP tools return correct response types | MCP protocol validation |
| AC-TEST-011-05 | Steering feedback updates neuromodulator state | State inspection after feedback |
| AC-TEST-011-06 | Formal verification integrates with immune system | Quarantine verification |

#### 3.11.5 Verification Method

```bash
# Run all Marblestone feature tests
cargo nextest run --workspace --test '*marblestone*' \
    --status-level all

# Run Marblestone integration tests
cargo nextest run --workspace --test 'marblestone_integration' \
    --retries 0

# Run Marblestone benchmarks
cargo bench --workspace --bench 'marblestone_*' -- \
    --sample-size 100 \
    --measurement-time 5

# Verify MCP tool responses
cargo run --bin mcp-test-harness -- \
    --tools get_steering_feedback,omni_infer,verify_code_node \
    --validate-schemas
```

---

## 4. Test Strategy

### 4.1 Test Pyramid

```
                    /\
                   /  \
                  /E2E \           <-- 5% - End-to-end MCP flows
                 /------\
                /        \
               / Integration\      <-- 15% - Cross-module tests
              /--------------\
             /                \
            /     Unit Tests   \   <-- 80% - Component isolation
           /____________________\
```

### 4.2 Test Data Management

| Data Type | Strategy | Storage |
|-----------|----------|---------|
| Unit Test Data | Inline fixtures | Code |
| Integration Test Data | Factory pattern | Generated |
| Performance Test Data | Production-like | Dedicated datasets |
| Chaos Test Data | Synthetic + Production | Mixed |

### 4.3 Test Environment Matrix

| Environment | Purpose | Data | Duration |
|-------------|---------|------|----------|
| CI | PR validation | Minimal | <30 min |
| Integration | Cross-module | Generated | <15 min |
| Performance | Benchmarks | Large datasets | 1-2 hours |
| Staging | Pre-production | Anonymized prod | Continuous |

---

## 5. Production Readiness Checklist

### 5.1 Pre-Deployment Checklist

- [ ] All unit tests pass with >90% coverage
- [ ] All integration tests pass with >80% coverage
- [ ] Performance benchmarks show no regressions (>5%)
- [ ] Load tests confirm capacity targets (10K concurrent)
- [ ] Chaos tests validate MTTR <30 seconds
- [ ] Security scan shows no high/critical vulnerabilities
- [ ] Dependency audit passes
- [ ] Docker image builds and scans clean
- [ ] Kubernetes manifests validated
- [ ] Monitoring dashboards configured
- [ ] Alert rules deployed
- [ ] Runbooks documented
- [ ] API documentation generated
- [ ] Changelog updated
- [ ] Version tagged

### 5.2 Deployment Verification

- [ ] Health checks pass within 30 seconds
- [ ] Metrics endpoint accessible
- [ ] Logs flowing to aggregator
- [ ] Traces appearing in tracing backend
- [ ] Smoke tests pass
- [ ] Rollback procedure verified

### 5.3 Post-Deployment Monitoring

| Metric | Threshold | Action |
|--------|-----------|--------|
| Error rate | >1% for 5m | Page on-call |
| P99 latency | >500ms for 5m | Alert |
| Availability | <99.9% | Page on-call |
| Memory usage | >80% | Alert |
| CPU usage | >90% | Alert |

---

## 6. Dependencies and Traceability

### 6.1 Module Dependency Matrix

| Requirement | Depends On | Tested By |
|-------------|------------|-----------|
| REQ-TEST-001 | Module 1 (Ghost) | Unit test framework |
| REQ-TEST-002 | Modules 1-13 | Integration harness |
| REQ-TEST-003 | Module 7 (CUDA), Module 8 (GPU-Direct) | Benchmarks |
| REQ-TEST-004 | Module 13 (MCP Hardening) | Load tester |
| REQ-TEST-005 | Module 11 (Immune) | Chaos engine |
| REQ-TEST-006 | Module 1 (Ghost) | Production config |
| REQ-TEST-007 | Modules 1-13 | Monitoring |
| REQ-TEST-008 | Module 13 (MCP Hardening) | Security tests |
| REQ-TEST-009 | All modules | Documentation |
| REQ-TEST-010 | All requirements | CI pipeline |
| REQ-TEST-011 | Modules 2, 4, 6, 9, 10, 11, 12, 13 | Marblestone feature tests |

### 6.2 PRD Traceability

| PRD Requirement | Module 14 Requirement | Status |
|-----------------|----------------------|--------|
| Quality Assurance | REQ-TEST-001, REQ-TEST-002 | Covered |
| Performance Targets | REQ-TEST-003, REQ-TEST-004 | Covered |
| Reliability | REQ-TEST-005 | Covered |
| Deployment | REQ-TEST-006 | Covered |
| Observability | REQ-TEST-007 | Covered |
| Security | REQ-TEST-008 | Covered |
| Documentation | REQ-TEST-009 | Covered |
| Automation | REQ-TEST-010 | Covered |
| Marblestone Features | REQ-TEST-011 | Covered |

### 6.3 Marblestone Feature Cross-Module Test References

The Marblestone feature tests (REQ-TEST-011) integrate with multiple module test suites:

| Module | Related Test IDs | Test Area |
|--------|------------------|-----------|
| Module 2 (Core Infrastructure) | T-MARB-001, T-MARB-007 | NeurotransmitterWeights struct, SteeringReward type |
| Module 4 (Knowledge Graph) | T-MARB-001-04, T-MARB-004, T-MARB-006 | Edge modulation, path analysis, bidirectional traversal |
| Module 6 (Bio-Nervous System) | T-MARB-002 | Lifecycle stages and lambda weight transitions |
| Module 9 (Dream Layer) | T-MARB-004, T-MARB-INT-005 | Memory consolidation and shortcut creation |
| Module 10 (Neuromodulation) | T-MARB-002, T-MARB-005, T-MARB-007 | Lambda adaptation, dopamine feedback, evaluation subsystems |
| Module 11 (Immune System) | T-MARB-003, T-MARB-INT-004 | Code verification and quarantine integration |
| Module 12 (Active Inference) | T-MARB-005, T-MARB-006, T-MARB-007 | Steering integration, OmniInferenceEngine |
| Module 13 (MCP Hardening) | T-MARB-003, T-MARB-008 | Verification boundaries, MCP tool security |

---

**Document Control**: Version 1.1.0 | 2025-12-31 | Specification Agent | Added Marblestone feature test requirements
