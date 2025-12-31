# Module 14: Testing & Production - Atomic Tasks

```yaml
metadata:
  module_id: "module-14"
  module_name: "Testing & Production Infrastructure"
  version: "1.0.0"
  phase: 13
  total_tasks: 35
  approach: "inside-out-bottom-up"
  created: "2025-12-31"
  dependencies:
    - module-01-ghost-system
    - module-02-core-infrastructure
    - module-03-embedding-pipeline
    - module-04-knowledge-graph
    - module-05-utl-integration
    - module-06-bio-nervous-system
    - module-07-cuda-optimization
    - module-08-gpu-direct-storage
    - module-09-dream-layer
    - module-10-neuromodulation
    - module-11-immune-system
    - module-12-active-inference
    - module-13-mcp-hardening
  estimated_duration: "3-4 weeks"
  spec_refs:
    - SPEC-TEST-014 (Functional)
    - TECH-TEST-014 (Technical)
```

---

## Task Overview

This module provides comprehensive testing infrastructure and production deployment capabilities for the ContextGraph system. It establishes the unit testing framework with >90% coverage, integration test harness with >80% cross-module coverage, performance benchmarks with <2% variance, load testing for 10K concurrent sessions, chaos engineering with MTTR <30s, and production-ready deployment configurations.

### Performance Targets

| Component | Target | Constraint |
|-----------|--------|------------|
| Unit Test Suite | <5 minutes | Full workspace execution |
| Integration Tests | <15 minutes | Cross-module scenarios |
| Benchmark Variance | <2% | Criterion statistical analysis |
| Load Test Capacity | 10,000 concurrent | Sustained throughput |
| Mean Time to Recovery | <30 seconds | Chaos fault injection |
| Production Startup | <10 seconds | Container initialization |
| CI Pipeline Total | <30 minutes | All stages parallel |

### Task Organization

1. **Foundation Layer** (Tasks 1-10): Unit test framework, mock registry, coverage config, fixtures
2. **Logic Layer** (Tasks 11-20): Integration harness, benchmarks, load tester, chaos engine
3. **Surface Layer** (Tasks 21-35): Production config, monitoring, security tests, Marblestone tests, CI/CD

---

## Foundation Layer: Unit Test Framework

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Test Discovery and Configuration
  # ============================================================

  - id: "TASK-14-001"
    title: "Implement TestDiscoveryConfig and TestFilter Structs"
    description: |
      Implement test discovery configuration for the unit test framework:

      TestDiscoveryConfig fields:
      - test_roots: Vec<PathBuf> (directories to scan)
      - file_patterns: Vec<String> (e.g., "*_test.rs", "test_*.rs")
      - include_patterns: Vec<String> (module inclusion)
      - exclude_patterns: Vec<String> (module exclusion)
      - max_depth: usize (default: 10)

      TestFilter fields:
      - module: Option<String>
      - test_name: Option<String>
      - tags: Vec<String>
      - exclude_slow: bool

      Implement Default trait with standard Rust test conventions.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-testing/src/unit/discovery.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - None (foundation struct)
    output_deliverables:
      - TestDiscoveryConfig struct
      - TestFilter struct
      - Default implementations
    acceptance_criteria:
      - "TestDiscoveryConfig struct with 5 fields compiles"
      - "TestFilter struct with 4 fields compiles"
      - "Default implementation uses '*_test.rs' and 'test_*.rs' patterns"
      - "max_depth defaults to 10"
      - "Clone, Debug, Serialize, Deserialize implemented"
    test_file: "crates/context-graph-testing/tests/discovery_tests.rs"

  - id: "TASK-14-002"
    title: "Implement CoverageConfig and CoverageTool Enum"
    description: |
      Implement coverage collection configuration:

      CoverageTool enum:
      - LlvmCov
      - Tarpaulin

      CoverageFormat enum:
      - Lcov
      - Html
      - Json
      - Cobertura

      CoverageConfig fields:
      - tool: CoverageTool (default: LlvmCov)
      - min_line_coverage: f64 (default: 90.0)
      - min_branch_coverage: f64 (default: 80.0)
      - min_function_coverage: f64 (default: 95.0)
      - include_paths: Vec<PathBuf>
      - exclude_paths: Vec<PathBuf>
      - output_format: CoverageFormat
      - output_path: PathBuf

      Implement validate() method checking coverage thresholds are in [0, 100].
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-testing/src/unit/coverage.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - None (foundation struct)
    output_deliverables:
      - CoverageTool enum
      - CoverageFormat enum
      - CoverageConfig struct
      - validate() method
    acceptance_criteria:
      - "CoverageTool enum with 2 variants"
      - "CoverageFormat enum with 4 variants"
      - "CoverageConfig struct with 8 fields"
      - "min_line_coverage defaults to 90.0"
      - "validate() returns error for thresholds outside [0, 100]"
      - "Clone, Debug, Serialize, Deserialize implemented"
    test_file: "crates/context-graph-testing/tests/coverage_config_tests.rs"

  - id: "TASK-14-003"
    title: "Implement ExecutionConfig and TestOrder Enum"
    description: |
      Implement test execution configuration:

      TestOrder enum:
      - Alphabetical
      - Random { seed: u64 }
      - Dependency

      ExecutionConfig fields:
      - max_threads: usize (default: num_cpus::get())
      - test_timeout: Duration (default: 30s)
      - suite_timeout: Duration (default: 5min)
      - retry_count: usize (default: 0)
      - fail_fast: bool (default: false)
      - order: TestOrder (default: Alphabetical)
      - capture_output: bool (default: true)
      - nocapture_on_failure: bool (default: true)

      Constraint: suite_timeout must be >= test_timeout * expected_tests.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 1.5
    file_path: "crates/context-graph-testing/src/unit/execution.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - None (foundation struct)
    output_deliverables:
      - TestOrder enum
      - ExecutionConfig struct
      - Default implementation
    acceptance_criteria:
      - "TestOrder enum with 3 variants"
      - "ExecutionConfig struct with 8 fields"
      - "max_threads defaults to available CPUs"
      - "suite_timeout defaults to 5 minutes"
      - "Clone, Debug implemented"
    test_file: "crates/context-graph-testing/tests/execution_config_tests.rs"

  - id: "TASK-14-004"
    title: "Implement MockRegistry and MockProvider Trait"
    description: |
      Implement mock registry for dependency injection:

      MockProvider trait:
      - fn name(&self) -> &str
      - fn create_instance(&self) -> Box<dyn Any + Send + Sync>
      - fn reset(&self)

      MockCall struct:
      - method: String
      - args: Vec<Box<dyn Any + Send + Sync>>
      - timestamp: Instant

      MockExpectation struct:
      - method: String
      - expected_args: Option<Vec<Box<dyn Any + Send + Sync>>>
      - return_value: Option<Box<dyn Any + Send + Sync>>
      - times: ExpectationTimes (Once, Times(n), AtLeast(n), Any)

      MockRegistry struct:
      - mocks: HashMap<String, Box<dyn MockProvider>>
      - call_records: HashMap<String, Vec<MockCall>>
      - expectations: HashMap<String, Vec<MockExpectation>>

      Methods:
      - register<T: MockProvider>(&mut self, provider: T)
      - get<T>(&self, name: &str) -> Option<&T>
      - record_call(&mut self, name: &str, call: MockCall)
      - verify_expectations(&self) -> Result<(), MockError>
    layer: "foundation"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/unit/mock.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - None (foundation struct)
    output_deliverables:
      - MockProvider trait
      - MockCall struct
      - MockExpectation struct
      - MockRegistry struct
      - ExpectationTimes enum
    acceptance_criteria:
      - "MockProvider trait with 3 methods"
      - "MockRegistry stores and retrieves providers"
      - "Call recording captures method, args, timestamp"
      - "verify_expectations() checks all expectations met"
      - "Returns MockError::UnmetExpectation for unmet expectations"
      - "Thread-safe (Send + Sync bounds)"
    test_file: "crates/context-graph-testing/tests/mock_registry_tests.rs"

  - id: "TASK-14-005"
    title: "Implement TestFixture Trait and FixtureManager"
    description: |
      Implement test fixture management:

      FixtureData struct:
      - data: HashMap<String, Box<dyn Any + Send + Sync>>
      - created_at: Instant

      FixtureError enum:
      - CreationFailed(String)
      - TeardownFailed(String)
      - Timeout
      - DependencyFailed(String)

      TestFixture trait:
      - fn name(&self) -> &str
      - async fn setup(&self) -> Result<FixtureData, FixtureError>
      - async fn teardown(&self, data: FixtureData) -> Result<(), FixtureError>
      - fn dependencies(&self) -> Vec<&str> { vec![] }

      FixtureInstance struct:
      - fixture_name: String
      - data: FixtureData
      - created_at: Instant

      FixtureManager struct:
      - fixtures: HashMap<String, Box<dyn TestFixture>>
      - dependencies: HashMap<String, Vec<String>>
      - active: Arc<RwLock<HashMap<String, FixtureInstance>>>
      - creation_timeout: Duration (default: 30s)

      Methods:
      - register(&mut self, fixture: Box<dyn TestFixture>)
      - async setup(&self, name: &str) -> Result<FixtureInstance, FixtureError>
      - async teardown(&self, name: &str) -> Result<(), FixtureError>
      - async teardown_all(&self) -> Vec<Result<(), FixtureError>>
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/unit/fixture.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-001"
      - "REQ-TEST-002"
    input_dependencies:
      - None (foundation struct)
    output_deliverables:
      - TestFixture trait
      - FixtureData struct
      - FixtureError enum
      - FixtureManager struct
    acceptance_criteria:
      - "TestFixture trait with 4 methods (1 default)"
      - "FixtureManager resolves fixture dependencies in topological order"
      - "Teardown is called for all active fixtures on test end"
      - "Timeout enforced on fixture creation"
      - "FixtureError provides actionable error messages"
    test_file: "crates/context-graph-testing/tests/fixture_tests.rs"

  - id: "TASK-14-006"
    title: "Implement FlakyTestDetector with Statistical Analysis"
    description: |
      Implement flaky test detection:

      TestOutcome enum:
      - Pass { duration: Duration }
      - Fail { error: String, duration: Duration }
      - Skip { reason: String }
      - Timeout { after: Duration }

      FlakyTestDetector struct:
      - history: HashMap<String, Vec<TestOutcome>>
      - threshold: f64 (default: 0.01, 1% flakiness)
      - min_runs: usize (default: 10)
      - window_size: usize (default: 100)

      FlakyTestReport struct:
      - test_name: String
      - failure_rate: f64
      - total_runs: usize
      - recent_failures: Vec<String>
      - is_flaky: bool

      Methods:
      - record_outcome(&mut self, test_name: &str, outcome: TestOutcome)
      - is_flaky(&self, test_name: &str) -> bool
      - get_failure_rate(&self, test_name: &str) -> Option<f64>
      - get_flaky_tests(&self) -> Vec<FlakyTestReport>
      - prune_old_history(&mut self)
    layer: "foundation"
    priority: "high"
    estimated_hours: 2.5
    file_path: "crates/context-graph-testing/src/unit/flaky.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - None (foundation struct)
    output_deliverables:
      - TestOutcome enum
      - FlakyTestDetector struct
      - FlakyTestReport struct
    acceptance_criteria:
      - "FlakyTestDetector tracks test outcomes over time"
      - "is_flaky() returns true when failure_rate > threshold with sufficient data"
      - "Window size limits memory usage"
      - "Statistical significance requires min_runs data points"
      - "get_flaky_tests() returns all tests exceeding threshold"
    test_file: "crates/context-graph-testing/tests/flaky_detector_tests.rs"

  - id: "TASK-14-007"
    title: "Implement TestResults and TestFailure Structs"
    description: |
      Implement test result aggregation:

      TestFailure struct:
      - test_name: String
      - module: String
      - error_message: String
      - stack_trace: Option<String>
      - stdout: String
      - stderr: String
      - duration: Duration
      - location: Option<SourceLocation>

      SourceLocation struct:
      - file: PathBuf
      - line: u32
      - column: u32

      TestResults struct:
      - passed: usize
      - failed: usize
      - skipped: usize
      - duration: Duration
      - failures: Vec<TestFailure>
      - test_details: Vec<TestDetail>

      TestDetail struct:
      - name: String
      - module: String
      - outcome: TestOutcome
      - duration: Duration

      Implement Display for TestResults with summary output.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-testing/src/unit/results.rs"
    dependencies:
      - "TASK-14-006"
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - TestOutcome enum from TASK-14-006
    output_deliverables:
      - TestFailure struct
      - SourceLocation struct
      - TestResults struct
      - TestDetail struct
      - Display implementation
    acceptance_criteria:
      - "TestResults aggregates pass/fail/skip counts"
      - "TestFailure includes stack trace when available"
      - "SourceLocation parsed from test output"
      - "Display shows summary: 'X passed, Y failed, Z skipped in Ns'"
      - "Clone, Debug, Serialize, Deserialize implemented"
    test_file: "crates/context-graph-testing/tests/results_tests.rs"

  - id: "TASK-14-008"
    title: "Implement CoverageReport and UncoveredLine Structs"
    description: |
      Implement coverage reporting:

      UncoveredLine struct:
      - file: PathBuf
      - line_number: u32
      - content: String
      - context: Option<String>

      CoverageReport struct:
      - line_coverage: f64 (target: >90%)
      - branch_coverage: f64 (target: >85%)
      - function_coverage: f64 (target: >95%)
      - uncovered_lines: Vec<UncoveredLine>
      - file_coverage: HashMap<PathBuf, FileCoverage>
      - generated_at: DateTime<Utc>
      - tool: CoverageTool

      FileCoverage struct:
      - path: PathBuf
      - line_coverage: f64
      - lines_covered: u32
      - lines_total: u32
      - uncovered_lines: Vec<u32>

      Methods:
      - meets_threshold(&self, config: &CoverageConfig) -> bool
      - generate_summary(&self) -> String
      - worst_files(&self, n: usize) -> Vec<&FileCoverage>
    layer: "foundation"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-testing/src/unit/coverage_report.rs"
    dependencies:
      - "TASK-14-002"
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - CoverageTool from TASK-14-002
      - CoverageConfig from TASK-14-002
    output_deliverables:
      - UncoveredLine struct
      - CoverageReport struct
      - FileCoverage struct
    acceptance_criteria:
      - "CoverageReport contains line, branch, function coverage"
      - "meets_threshold() checks all coverage types against config"
      - "uncovered_lines populated with file context"
      - "worst_files() returns files with lowest coverage"
      - "Serialize, Deserialize for JSON output"
    test_file: "crates/context-graph-testing/tests/coverage_report_tests.rs"

  - id: "TASK-14-009"
    title: "Implement TestResultAggregator"
    description: |
      Implement test result aggregation:

      TestResultAggregator struct:
      - results: RwLock<Vec<TestResults>>
      - start_time: Instant
      - module_results: HashMap<String, ModuleTestResult>

      ModuleTestResult struct:
      - module: String
      - passed: usize
      - failed: usize
      - skipped: usize
      - duration: Duration

      Methods:
      - new() -> Self
      - add_result(&self, result: TestResults)
      - get_summary(&self) -> TestSummary
      - get_module_summary(&self, module: &str) -> Option<ModuleTestResult>
      - to_junit_xml(&self) -> String
      - to_json(&self) -> String

      TestSummary struct:
      - total_passed: usize
      - total_failed: usize
      - total_skipped: usize
      - total_duration: Duration
      - module_count: usize
      - flaky_count: usize
    layer: "foundation"
    priority: "high"
    estimated_hours: 2.5
    file_path: "crates/context-graph-testing/src/unit/aggregator.rs"
    dependencies:
      - "TASK-14-007"
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - TestResults from TASK-14-007
    output_deliverables:
      - TestResultAggregator struct
      - ModuleTestResult struct
      - TestSummary struct
      - JUnit XML export
    acceptance_criteria:
      - "TestResultAggregator is thread-safe (RwLock)"
      - "Aggregates results from multiple test runs"
      - "to_junit_xml() produces valid JUnit XML for CI"
      - "Module breakdown available via get_module_summary()"
      - "Total duration is wall-clock time, not sum"
    test_file: "crates/context-graph-testing/tests/aggregator_tests.rs"

  - id: "TASK-14-010"
    title: "Implement UnitTestFramework with Nextest Integration"
    description: |
      Implement the main unit test framework:

      UnitTestFramework struct:
      - discovery: TestDiscoveryConfig
      - mock_registry: MockRegistry
      - coverage: CoverageConfig
      - execution: ExecutionConfig
      - results: TestResultAggregator
      - flaky_detector: FlakyTestDetector

      Methods:
      - new(config: UnitTestConfig) -> Self
      - run_tests(&self, filter: TestFilter) -> Result<TestResults, TestError>
        [Constraint: <5 minutes for full suite]
      - collect_coverage(&self) -> Result<CoverageReport, CoverageError>
      - create_mock<T: Mockable>(&self, name: &str) -> Mock<T>
      - load_fixtures(&self, path: &Path) -> Result<FixtureData, FixtureError>
      - generate_report(&self, format: ReportFormat) -> String

      Nextest integration:
      - Execute via `cargo nextest run` subprocess
      - Parse nextest JSON output for structured results
      - Support nextest profiles (default, ci, coverage)

      Constraint: Total unit test execution <5 minutes.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/unit/framework.rs"
    dependencies:
      - "TASK-14-001"
      - "TASK-14-002"
      - "TASK-14-003"
      - "TASK-14-004"
      - "TASK-14-005"
      - "TASK-14-006"
      - "TASK-14-009"
    requirements_traced:
      - "REQ-TEST-001"
    input_dependencies:
      - All foundation structs from TASK-14-001 through TASK-14-009
    output_deliverables:
      - UnitTestFramework struct
      - Nextest JSON output parser
      - Coverage integration
    acceptance_criteria:
      - "UnitTestFramework struct with 6 fields"
      - "run_tests() executes nextest and parses output"
      - "collect_coverage() runs llvm-cov and parses lcov output"
      - "Full test suite completes in <5 minutes"
      - "Filter supports module, name, and tag filtering"
      - "Parallel execution with configurable threads"
    test_file: "crates/context-graph-testing/tests/framework_tests.rs"

  # ============================================================
  # LOGIC LAYER: Integration Test Harness
  # ============================================================

  - id: "TASK-14-011"
    title: "Implement IsolationManager with Database Strategies"
    description: |
      Implement test isolation management:

      DatabaseIsolation enum:
      - SeparateDatabase
      - TransactionRollback
      - SnapshotRestore

      FilesystemIsolation struct:
      - temp_dir: Option<TempDir>
      - overlay_mounts: Vec<PathBuf>

      NetworkIsolation struct:
      - port_range: Range<u16>
      - allocated_ports: HashSet<u16>

      MemoryIsolation struct:
      - sandboxed: bool
      - memory_limit: Option<usize>

      IsolationManager struct:
      - database_strategy: DatabaseIsolation
      - filesystem_strategy: FilesystemIsolation
      - network_strategy: NetworkIsolation
      - memory_strategy: MemoryIsolation

      Methods:
      - async setup_isolation(&mut self) -> Result<IsolationContext, IsolationError>
      - async teardown_isolation(&mut self, ctx: IsolationContext) -> Result<()>
      - allocate_port(&mut self) -> u16
      - get_temp_dir(&self) -> &Path
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/integration/isolation.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-002"
    input_dependencies:
      - None (logic layer foundation)
    output_deliverables:
      - DatabaseIsolation enum
      - IsolationManager struct
      - IsolationContext struct
    acceptance_criteria:
      - "SeparateDatabase creates new database per test"
      - "TransactionRollback wraps test in transaction, rolls back after"
      - "Filesystem isolation uses temp directories"
      - "Network isolation allocates unique ports"
      - "All isolation contexts cleaned up on teardown"
    test_file: "crates/context-graph-testing/tests/isolation_tests.rs"

  - id: "TASK-14-012"
    title: "Implement ModuleOrchestrator for Multi-Component Tests"
    description: |
      Implement module orchestration:

      ModuleId enum: (Core, Embedding, Graph, Memory, Query, Cuda, Storage, etc.)

      ModuleConnection struct:
      - from: ModuleId
      - to: ModuleId
      - connection_type: ConnectionType

      ConnectionType enum:
      - Direct
      - AsyncChannel
      - SharedMemory

      HealthCheck trait:
      - async fn check(&self) -> Result<HealthStatus, HealthError>

      ModuleOrchestrator struct:
      - modules: HashMap<ModuleId, Arc<dyn ModuleInstance>>
      - connections: Vec<ModuleConnection>
      - health_checks: HashMap<ModuleId, Box<dyn HealthCheck>>
      - startup_order: Vec<ModuleId>

      Methods:
      - new() -> Self
      - register_module(&mut self, id: ModuleId, instance: Arc<dyn ModuleInstance>)
      - async start_all(&self) -> Result<(), OrchestrationError>
      - async stop_all(&self) -> Result<()>
      - async health_check_all(&self) -> HashMap<ModuleId, HealthStatus>
      - get_module<T>(&self, id: ModuleId) -> Option<Arc<T>>
    layer: "logic"
    priority: "critical"
    estimated_hours: 3.5
    file_path: "crates/context-graph-testing/src/integration/orchestrator.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-002"
    input_dependencies:
      - None (logic layer foundation)
    output_deliverables:
      - ModuleId enum
      - ModuleOrchestrator struct
      - ModuleConnection struct
      - HealthCheck trait
    acceptance_criteria:
      - "ModuleId covers all 13 modules"
      - "Startup order respects module dependencies"
      - "Health checks run in parallel"
      - "Stop order is reverse of startup order"
      - "Module retrieval is type-safe"
    test_file: "crates/context-graph-testing/tests/orchestrator_tests.rs"

  - id: "TASK-14-013"
    title: "Implement IntegrationTestSuite and IntegrationTestCase"
    description: |
      Implement integration test definitions:

      TestAction enum:
      - CreateContext { spec: ContextSpec }
      - QueryContext { query: String, expected: usize }
      - UpdateContext { id: Uuid, delta: String }
      - DeleteContext { id: Uuid }
      - WaitForSync { timeout: Duration }
      - InjectFault { fault: FaultSpec }
      - Custom { name: String, args: serde_json::Value }

      TestAssertion enum:
      - Equals { path: String, value: serde_json::Value }
      - Contains { path: String, substring: String }
      - GreaterThan { path: String, value: f64 }
      - LessThan { path: String, value: f64 }
      - Exists { path: String }
      - Custom { name: String, validator: Box<dyn Fn(&serde_json::Value) -> bool> }

      IntegrationTestCase struct:
      - id: String
      - description: String
      - modules: Vec<ModuleId>
      - setup: Vec<TestAction>
      - actions: Vec<TestAction>
      - assertions: Vec<TestAssertion>
      - cleanup: Vec<TestAction>
      - expected_duration: Duration

      IntegrationTestSuite struct:
      - id: String
      - name: String
      - required_modules: Vec<ModuleId>
      - required_fixtures: Vec<String>
      - test_cases: Vec<IntegrationTestCase>
      - timeout: Duration
      - parallel: bool
    layer: "logic"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/integration/suite.rs"
    dependencies:
      - "TASK-14-012"
    requirements_traced:
      - "REQ-TEST-002"
    input_dependencies:
      - ModuleId from TASK-14-012
    output_deliverables:
      - TestAction enum
      - TestAssertion enum
      - IntegrationTestCase struct
      - IntegrationTestSuite struct
    acceptance_criteria:
      - "TestAction covers all common integration operations"
      - "TestAssertion supports JSON path-based validation"
      - "Custom actions/assertions allow extension"
      - "Suite can be serialized to YAML for external definition"
      - "Parallel flag controls concurrent test execution"
    test_file: "crates/context-graph-testing/tests/suite_tests.rs"

  - id: "TASK-14-014"
    title: "Implement IntegrationResultCollector and CleanupTracker"
    description: |
      Implement integration test result collection:

      StepResult struct:
      - step: TestAction
      - success: bool
      - duration: Duration
      - output: Option<serde_json::Value>
      - error: Option<String>

      ScenarioResult struct:
      - scenario_id: String
      - success: bool
      - step_results: Vec<StepResult>
      - duration: Duration
      - resource_usage: ResourceMetrics

      ResourceMetrics struct:
      - peak_memory_mb: f64
      - avg_cpu_percent: f64
      - network_bytes_sent: u64
      - network_bytes_recv: u64
      - disk_io_bytes: u64

      IntegrationResultCollector struct:
      - results: RwLock<Vec<ScenarioResult>>
      - resource_monitor: ResourceMonitor

      CleanupTracker struct:
      - created_resources: Vec<CleanupAction>
      - cleanup_order: Vec<usize>

      CleanupAction enum:
      - DeleteContext(Uuid)
      - StopService(String)
      - RemoveFile(PathBuf)
      - RollbackTransaction(String)
      - Custom(Box<dyn Fn() -> Result<()>>)
    layer: "logic"
    priority: "high"
    estimated_hours: 2.5
    file_path: "crates/context-graph-testing/src/integration/collector.rs"
    dependencies:
      - "TASK-14-013"
    requirements_traced:
      - "REQ-TEST-002"
    input_dependencies:
      - TestAction from TASK-14-013
    output_deliverables:
      - StepResult struct
      - ScenarioResult struct
      - ResourceMetrics struct
      - IntegrationResultCollector struct
      - CleanupTracker struct
    acceptance_criteria:
      - "StepResult captures action outcome with timing"
      - "ResourceMetrics collected during scenario execution"
      - "CleanupTracker executes cleanup in reverse order"
      - "All cleanup actions executed even if some fail"
      - "Thread-safe result collection"
    test_file: "crates/context-graph-testing/tests/collector_tests.rs"

  - id: "TASK-14-015"
    title: "Implement IntegrationTestHarness"
    description: |
      Implement the main integration test harness:

      IntegrationTestHarness struct:
      - fixtures: FixtureManager
      - orchestrator: ModuleOrchestrator
      - isolation: IsolationManager
      - test_suites: Vec<IntegrationTestSuite>
      - results: IntegrationResultCollector
      - cleanup: CleanupTracker

      TestEnvironment struct:
      - isolation_context: IsolationContext
      - active_modules: Vec<ModuleId>
      - fixtures_loaded: Vec<String>

      Methods:
      - new() -> Self
      - async setup(&mut self, config: TestConfig) -> Result<TestEnvironment>
      - async run_scenario(&self, scenario: &IntegrationTestCase) -> ScenarioResult
        [Constraint: <15 minutes for full suite]
      - async teardown(&mut self) -> Result<()>
      - inject_network_condition(&self, condition: NetworkCondition)
      - async run_all_suites(&self) -> Vec<ScenarioResult>

      Constraint: Integration test execution <15 minutes.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/integration/harness.rs"
    dependencies:
      - "TASK-14-005"
      - "TASK-14-011"
      - "TASK-14-012"
      - "TASK-14-013"
      - "TASK-14-014"
    requirements_traced:
      - "REQ-TEST-002"
    input_dependencies:
      - FixtureManager from TASK-14-005
      - IsolationManager from TASK-14-011
      - ModuleOrchestrator from TASK-14-012
      - IntegrationTestSuite from TASK-14-013
      - IntegrationResultCollector from TASK-14-014
    output_deliverables:
      - IntegrationTestHarness struct
      - TestEnvironment struct
      - Cross-module scenario execution
    acceptance_criteria:
      - "IntegrationTestHarness struct with 6 fields"
      - "setup() initializes isolation and starts modules"
      - "run_scenario() executes actions and validates assertions"
      - "teardown() cleans up all resources"
      - "Full integration suite completes in <15 minutes"
      - ">80% cross-module coverage verified"
    test_file: "crates/context-graph-testing/tests/harness_tests.rs"

  # ============================================================
  # LOGIC LAYER: Performance Benchmarks
  # ============================================================

  - id: "TASK-14-016"
    title: "Implement BenchmarkDefinition and StatisticalConfig"
    description: |
      Implement benchmark configuration:

      BenchmarkCategory enum:
      - Latency
      - Throughput
      - Memory
      - CPUCycles
      - CacheHits
      - IOOperations

      BenchmarkDefinition struct:
      - id: String
      - category: BenchmarkCategory
      - module: ModuleId
      - operation: String
      - expected_latency: Option<Duration>
      - expected_throughput: Option<f64>
      - warmup_iterations: usize (default: 100)
      - measurement_iterations: usize (default: 1000)
      - sample_size: usize (default: 100)

      StatisticalConfig struct:
      - confidence_level: f64 (default: 0.95)
      - noise_threshold: f64 (default: 0.02, 2%)
      - outlier_sensitivity: f64 (default: 3.0, 3 sigma)
      - min_samples: usize (default: 100)
      - max_relative_error: f64 (default: 0.02, 2%)

      Constraint: Benchmark variance <2%.
    layer: "logic"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-testing/src/bench/definition.rs"
    dependencies:
      - "TASK-14-012"
    requirements_traced:
      - "REQ-TEST-003"
    input_dependencies:
      - ModuleId from TASK-14-012
    output_deliverables:
      - BenchmarkCategory enum
      - BenchmarkDefinition struct
      - StatisticalConfig struct
    acceptance_criteria:
      - "BenchmarkCategory covers 6 measurement types"
      - "BenchmarkDefinition has module association"
      - "StatisticalConfig has 2% noise threshold"
      - "Warmup separates from measurement"
      - "Clone, Debug, Serialize, Deserialize implemented"
    test_file: "crates/context-graph-testing/tests/bench_definition_tests.rs"

  - id: "TASK-14-017"
    title: "Implement BaselineRegistry with Regression Detection"
    description: |
      Implement baseline storage and comparison:

      BaselineMeasurement struct:
      - mean_ns: f64
      - std_dev_ns: f64
      - median_ns: f64
      - p99_ns: f64
      - throughput: f64
      - timestamp: i64
      - commit: String
      - samples: usize

      StatisticalTest enum:
      - TTest
      - MannWhitneyU
      - KolmogorovSmirnov

      RegressionResult struct:
      - benchmark_id: String
      - baseline: BaselineMeasurement
      - current: BaselineMeasurement
      - change_percent: f64
      - is_regression: bool
      - p_value: f64
      - confidence: f64

      RegressionDetector struct:
      - test_type: StatisticalTest
      - significance_level: f64 (default: 0.05)
      - min_effect_size: f64 (default: 0.05, 5%)

      BaselineRegistry struct:
      - baselines: HashMap<String, BaselineMeasurement>
      - storage_path: PathBuf
      - version: String

      Methods:
      - load(&mut self, path: &Path) -> Result<()>
      - save(&self, path: &Path) -> Result<()>
      - update_baseline(&mut self, id: &str, measurement: BaselineMeasurement)
      - detect_regression(&self, id: &str, current: &BaselineMeasurement) -> Option<RegressionResult>
    layer: "logic"
    priority: "critical"
    estimated_hours: 3.5
    file_path: "crates/context-graph-testing/src/bench/baseline.rs"
    dependencies:
      - "TASK-14-016"
    requirements_traced:
      - "REQ-TEST-003"
    input_dependencies:
      - BenchmarkDefinition from TASK-14-016
    output_deliverables:
      - BaselineMeasurement struct
      - BaselineRegistry struct
      - RegressionDetector struct
      - RegressionResult struct
    acceptance_criteria:
      - "BaselineRegistry persists to JSON file"
      - "detect_regression() uses configured statistical test"
      - "Regression detected with 95% confidence threshold"
      - "Change percent calculated correctly"
      - "Git commit hash stored for traceability"
    test_file: "crates/context-graph-testing/tests/baseline_tests.rs"

  - id: "TASK-14-018"
    title: "Implement PerformanceBenchmarks with Criterion Integration"
    description: |
      Implement the performance benchmark suite:

      ResourceMonitor struct:
      - cpu_usage: AtomicU64
      - memory_usage: AtomicU64
      - sampling_interval: Duration

      BenchmarkResultStore struct:
      - results: HashMap<String, Vec<BenchmarkResult>>
      - storage_path: PathBuf

      BenchmarkResult struct:
      - id: String
      - measurement: BaselineMeasurement
      - resource_usage: ResourceMetrics
      - timestamp: DateTime<Utc>

      PerformanceBenchmarks struct:
      - benchmarks: HashMap<String, BenchmarkDefinition>
      - baselines: BaselineRegistry
      - stats_config: StatisticalConfig
      - resource_monitor: ResourceMonitor
      - results: BenchmarkResultStore
      - regression_detector: RegressionDetector

      Methods:
      - new(config: BenchmarkConfig) -> Self
      - register_benchmark(&mut self, def: BenchmarkDefinition)
      - run_all(&self) -> Vec<BenchmarkResult>
        [Constraint: <2% variance]
      - run_benchmark(&self, id: &str) -> Option<BenchmarkResult>
      - check_regressions(&self) -> Vec<RegressionResult>
      - export_report(&self, format: ReportFormat) -> String

      Criterion integration:
      - Use criterion::Criterion for measurement
      - Configure sample_size, measurement_time, warmup_time
      - Parse criterion JSON output for structured results

      Constraint: Benchmark variance <2%.
    layer: "logic"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/bench/framework.rs"
    dependencies:
      - "TASK-14-016"
      - "TASK-14-017"
    requirements_traced:
      - "REQ-TEST-003"
    input_dependencies:
      - BenchmarkDefinition from TASK-14-016
      - BaselineRegistry from TASK-14-017
    output_deliverables:
      - PerformanceBenchmarks struct
      - BenchmarkResultStore struct
      - ResourceMonitor struct
      - Criterion integration
    acceptance_criteria:
      - "PerformanceBenchmarks struct with 6 fields"
      - "run_all() executes all registered benchmarks"
      - "Variance is <2% across runs"
      - "Resource usage captured during benchmarks"
      - "check_regressions() compares against baselines"
      - "Report export in HTML, JSON formats"
    test_file: "crates/context-graph-testing/tests/benchmarks_tests.rs"

  # ============================================================
  # LOGIC LAYER: Load Testing
  # ============================================================

  - id: "TASK-14-019"
    title: "Implement LoadProfile and WorkloadScenarios"
    description: |
      Implement load testing configuration:

      LoadProfileType enum:
      - Constant
      - Ramp
      - Step { step_size: usize, step_duration: Duration }
      - Spike { spike_multiplier: f64, spike_duration: Duration }
      - Stress { increment: usize, threshold_errors: f64 }

      ThinkTime enum:
      - Fixed(Duration)
      - Uniform { min: Duration, max: Duration }
      - Exponential { mean: Duration }
      - None

      LoadProfile struct:
      - profile_type: LoadProfileType
      - target_rps: u64
      - max_concurrent: usize (target: 10,000)
      - ramp_up: Duration
      - sustained: Duration
      - ramp_down: Duration
      - think_time: ThinkTime

      RequestType enum:
      - Read
      - Write
      - ReadWrite
      - Compute

      LoadRequest struct:
      - request_type: RequestType
      - tool: String (MCP tool name)
      - params: serde_json::Value
      - expected_latency: Duration
      - timeout: Duration
      - validation: Vec<ResponseValidation>

      LoadScenario struct:
      - id: String
      - name: String
      - weight: f64
      - requests: Vec<LoadRequest>
      - success_criteria: SuccessCriteria

      Constraint: Support 10,000 concurrent sessions.
    layer: "logic"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/load/profile.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-004"
    input_dependencies:
      - None (logic layer foundation)
    output_deliverables:
      - LoadProfileType enum
      - LoadProfile struct
      - LoadRequest struct
      - LoadScenario struct
    acceptance_criteria:
      - "LoadProfile supports 5 profile types"
      - "max_concurrent can be set to 10,000"
      - "ThinkTime configures inter-request delays"
      - "LoadScenario has weighted request distribution"
      - "Serialize/Deserialize for YAML scenario files"
    test_file: "crates/context-graph-testing/tests/load_profile_tests.rs"

  - id: "TASK-14-020"
    title: "Implement LoadMetricsCollector with HDR Histogram"
    description: |
      Implement real-time load test metrics:

      LoadMetricsCollector struct:
      - rps: AtomicU64
      - active_connections: AtomicU64
      - latency_histogram: Arc<RwLock<hdrhistogram::Histogram<u64>>>
      - errors: DashMap<String, AtomicU64>
      - throughput_bytes: AtomicU64
      - interval: Duration
      - time_series: Arc<RwLock<Vec<MetricsSnapshot>>>

      MetricsSnapshot struct:
      - timestamp: Instant
      - rps: u64
      - active_connections: u64
      - latency_p50: Duration
      - latency_p95: Duration
      - latency_p99: Duration
      - latency_p999: Duration
      - error_rate: f64
      - throughput_bytes: u64

      SuccessCriteria struct:
      - max_error_rate: f64 (default: 0.01, 1%)
      - max_p99_latency: Duration
      - min_throughput: f64
      - max_saturation: f64

      SaturationDetector struct:
      - queue_depth: AtomicU64
      - latency_trend: Vec<f64>
      - error_trend: Vec<f64>
      - saturation_threshold: f64

      Methods:
      - new() -> Self
      - record_request(&self, latency: Duration, success: bool)
      - snapshot(&self) -> MetricsSnapshot
      - percentile(&self, p: f64) -> Duration
      - is_saturated(&self) -> bool
      - meets_criteria(&self, criteria: &SuccessCriteria) -> bool
    layer: "logic"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/load/metrics.rs"
    dependencies:
      - "TASK-14-019"
    requirements_traced:
      - "REQ-TEST-004"
    input_dependencies:
      - SuccessCriteria from TASK-14-019
    output_deliverables:
      - LoadMetricsCollector struct
      - MetricsSnapshot struct
      - SaturationDetector struct
    acceptance_criteria:
      - "HDR Histogram provides accurate percentiles"
      - "Time series snapshots at configurable interval"
      - "Saturation detection via queue depth and latency trend"
      - "Atomic counters for thread-safe updates"
      - "P50, P95, P99, P99.9 available"
    test_file: "crates/context-graph-testing/tests/load_metrics_tests.rs"

  - id: "TASK-14-021"
    title: "Implement LoadTester with 10K Concurrent Sessions"
    description: |
      Implement the load testing framework:

      ClientPool struct:
      - capacity: usize
      - active: Arc<Semaphore>
      - factory: Box<dyn ClientFactory>
      - connection_reuse: bool

      ClientFactory trait:
      - fn create(&self) -> Box<dyn Client>

      LoadTestResult struct:
      - total_requests: u64
      - successful_requests: u64
      - failed_requests: u64
      - latency_p50: Duration
      - latency_p95: Duration
      - latency_p99: Duration
      - throughput: f64
      - error_rate: f64
      - resource_usage: ResourceUsage
      - saturation_point: Option<u64>

      BreakpointResult struct:
      - max_sustainable_rps: u64
      - breaking_rps: u64
      - p99_at_break: Duration
      - error_rate_at_break: f64

      LoadTester struct:
      - profile: LoadProfile
      - client_pool: ClientPool
      - metrics: LoadMetricsCollector
      - saturation_detector: SaturationDetector
      - reporter: LoadTestReporter
      - scenarios: Vec<LoadScenario>

      Methods:
      - new(profile: LoadProfile) -> Self
      - configure(&mut self, config: LoadConfig) -> Result<()>
      - async run(&self) -> LoadTestResult
        [Constraint: 10K concurrent]
      - get_metrics(&self) -> LiveMetrics
      - async find_breakpoint(&self) -> BreakpointResult

      Constraint: Support 10,000 concurrent sessions.
    layer: "logic"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/load/tester.rs"
    dependencies:
      - "TASK-14-019"
      - "TASK-14-020"
    requirements_traced:
      - "REQ-TEST-004"
    input_dependencies:
      - LoadProfile from TASK-14-019
      - LoadMetricsCollector from TASK-14-020
    output_deliverables:
      - LoadTester struct
      - ClientPool struct
      - LoadTestResult struct
      - BreakpointResult struct
    acceptance_criteria:
      - "LoadTester supports 10,000 concurrent sessions"
      - "Client pool manages connection reuse"
      - "find_breakpoint() identifies system capacity limit"
      - "Real-time metrics available during test"
      - "Graceful ramp-up and ramp-down"
    test_file: "crates/context-graph-testing/tests/load_tester_tests.rs"

  # ============================================================
  # LOGIC LAYER: Chaos Engineering
  # ============================================================

  - id: "TASK-14-022"
    title: "Implement FaultType Enum and FaultRegistry"
    description: |
      Implement fault injection types:

      ProcessTarget enum:
      - ByName(String)
      - ByPid(u32)
      - Random { count: usize }

      NetworkTarget struct:
      - host: String
      - port: Option<u16>

      DiskFailureType enum:
      - ReadOnly
      - Full
      - Slow { latency: Duration }
      - Corrupt { rate: f64 }

      CorruptionType enum:
      - BitFlip { rate: f64 }
      - Truncate { bytes: usize }
      - RandomBytes { count: usize }

      FaultType enum:
      - ProcessCrash { target: ProcessTarget, restart_delay: Option<Duration> }
      - NetworkPartition { partitions: Vec<Vec<String>>, duration: Duration }
      - NetworkLatency { target: NetworkTarget, latency: Duration, jitter: Duration }
      - PacketLoss { target: NetworkTarget, loss_rate: f64 }
      - DiskFailure { target: String, failure_type: DiskFailureType }
      - CPUStress { utilization: f64, duration: Duration }
      - MemoryPressure { allocation_mb: usize, duration: Duration }
      - ClockSkew { offset: Duration, direction: ClockDirection }
      - DependencyFailure { service: String, failure_mode: DependencyFailureMode }
      - DataCorruption { target: DataTarget, corruption_type: CorruptionType }

      FaultRegistry struct:
      - active_faults: HashMap<Uuid, FaultHandle>
      - fault_history: Vec<FaultRecord>

      FaultHandle struct:
      - id: Uuid
      - fault_type: FaultType
      - started_at: Instant
      - abort_tx: broadcast::Sender<()>
    layer: "logic"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/chaos/fault.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-005"
    input_dependencies:
      - None (logic layer foundation)
    output_deliverables:
      - FaultType enum (10 variants)
      - FaultRegistry struct
      - FaultHandle struct
    acceptance_criteria:
      - "FaultType covers 10 failure modes"
      - "FaultHandle allows abort during injection"
      - "FaultRegistry tracks active and historical faults"
      - "NetworkPartition splits nodes into isolated groups"
      - "All fault types are reproducible"
    test_file: "crates/context-graph-testing/tests/fault_tests.rs"

  - id: "TASK-14-023"
    title: "Implement ChaosExperiment and SafetyControls"
    description: |
      Implement chaos experiment definitions:

      FaultInjection struct:
      - fault: FaultType
      - delay: Duration (when to inject)
      - duration: Duration (how long to maintain)

      SteadyStateMetric struct:
      - name: String
      - min: f64
      - max: f64
      - tolerance: f64

      SteadyStateDefinition struct:
      - metrics: Vec<SteadyStateMetric>
      - check_interval: Duration
      - required_duration: Duration

      RecoveryExpectation struct:
      - max_recovery_time: Duration (target: <30s)
      - allow_data_loss: bool
      - allow_degradation: bool

      AbortCondition enum:
      - ErrorRateExceeds(f64)
      - LatencyExceeds(Duration)
      - DataCorruptionDetected
      - HealthCheckFailed
      - Custom(Box<dyn Fn() -> bool>)

      ChaosExperiment struct:
      - id: String
      - name: String
      - hypothesis: String
      - faults: Vec<FaultInjection>
      - duration: Duration
      - steady_state: SteadyStateDefinition
      - recovery: RecoveryExpectation
      - abort_conditions: Vec<AbortCondition>

      BlastRadiusConfig struct:
      - max_affected_pct: f64
      - max_affected_count: usize
      - protected: Vec<String>
      - min_healthy: usize

      SafetyControls struct:
      - emergency_stop: broadcast::Sender<()>
      - environment_guard: EnvironmentGuard
      - blast_radius: BlastRadiusConfig
      - auto_rollback: bool
      - max_duration: Duration

      Constraint: MTTR <30 seconds.
    layer: "logic"
    priority: "high"
    estimated_hours: 3.5
    file_path: "crates/context-graph-testing/src/chaos/experiment.rs"
    dependencies:
      - "TASK-14-022"
    requirements_traced:
      - "REQ-TEST-005"
    input_dependencies:
      - FaultType from TASK-14-022
    output_deliverables:
      - ChaosExperiment struct
      - SafetyControls struct
      - AbortCondition enum
      - SteadyStateDefinition struct
    acceptance_criteria:
      - "ChaosExperiment defines hypothesis and expected outcome"
      - "SafetyControls prevent runaway chaos"
      - "AbortCondition triggers automatic experiment termination"
      - "BlastRadiusConfig limits scope of impact"
      - "Emergency stop immediately halts all faults"
    test_file: "crates/context-graph-testing/tests/experiment_tests.rs"

  - id: "TASK-14-024"
    title: "Implement RecoveryValidator and ChaosEngine"
    description: |
      Implement chaos engine and recovery validation:

      RecoveryMetrics struct:
      - time_to_detect: Duration
      - time_to_recover: Duration (target: <30s)
      - data_loss: Option<usize>
      - service_degradation: f64
      - automatic_recovery: bool

      DataIntegrityChecker struct:
      - checksums: HashMap<String, u64>
      - verification_queries: Vec<String>

      RecoveryValidator struct:
      - expected_recovery: Duration
      - integrity_checker: DataIntegrityChecker
      - health_checker: ServiceHealthChecker
      - recovery_metrics: RecoveryMetrics

      ChaosMonitor struct:
      - metrics_collector: MetricsCollector
      - alert_channels: Vec<AlertChannel>

      ChaosScheduler struct:
      - scheduled_experiments: Vec<ScheduledExperiment>
      - cron_expressions: HashMap<String, String>

      ChaosEngine struct:
      - fault_registry: FaultRegistry
      - experiments: Vec<ChaosExperiment>
      - monitor: ChaosMonitor
      - recovery_validator: RecoveryValidator
      - safety: SafetyControls
      - scheduler: ChaosScheduler

      Methods:
      - new(config: ChaosConfig) -> Self
      - async inject_fault(&self, fault: FaultType) -> FaultHandle
      - async monitor_recovery(&self, handle: FaultHandle) -> RecoveryMetrics
        [Constraint: MTTR <30s]
      - async abort(&self, handle: FaultHandle) -> Result<()>
      - schedule_gameday(&self, schedule: GameDaySchedule) -> GameDayHandle
      - async run_experiment(&self, experiment: &ChaosExperiment) -> ExperimentResult

      Constraint: MTTR <30 seconds.
    layer: "logic"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/chaos/engine.rs"
    dependencies:
      - "TASK-14-022"
      - "TASK-14-023"
    requirements_traced:
      - "REQ-TEST-005"
    input_dependencies:
      - FaultRegistry from TASK-14-022
      - ChaosExperiment from TASK-14-023
      - SafetyControls from TASK-14-023
    output_deliverables:
      - ChaosEngine struct
      - RecoveryValidator struct
      - RecoveryMetrics struct
      - ChaosMonitor struct
    acceptance_criteria:
      - "ChaosEngine struct with 6 fields"
      - "inject_fault() returns handle for tracking"
      - "monitor_recovery() tracks MTTR"
      - "MTTR measured and validated <30s"
      - "Data integrity verified after recovery"
      - "Gameday scheduling for recurring chaos"
    test_file: "crates/context-graph-testing/tests/chaos_engine_tests.rs"

  # ============================================================
  # SURFACE LAYER: Production Configuration
  # ============================================================

  - id: "TASK-14-025"
    title: "Implement DockerConfig and Dockerfile Generation"
    description: |
      Implement Docker configuration:

      BuildStage struct:
      - name: String
      - from: String
      - commands: Vec<String>
      - copy_from: Option<String>

      PortMapping struct:
      - container_port: u16
      - host_port: Option<u16>
      - protocol: Protocol (TCP, UDP)

      DockerHealthCheck struct:
      - command: Vec<String>
      - interval: Duration
      - timeout: Duration
      - start_period: Duration
      - retries: u32

      DockerSecurity struct:
      - run_as_user: Option<String>
      - read_only_root: bool
      - no_new_privileges: bool
      - capabilities_drop: Vec<String>

      DockerConfig struct:
      - base_image: String (default: "rust:1.75-slim-bookworm")
      - build_stages: Vec<BuildStage>
      - runtime_image: String (default: "debian:bookworm-slim")
      - ports: Vec<PortMapping>
      - volumes: Vec<VolumeMount>
      - env: HashMap<String, String>
      - healthcheck: DockerHealthCheck
      - labels: HashMap<String, String>
      - security: DockerSecurity

      Methods:
      - generate_dockerfile(&self) -> String
      - generate_dockerignore(&self) -> String
      - validate(&self) -> Result<(), ConfigError>

      Constraint: Container startup <10 seconds.
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/production/docker.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-006"
    input_dependencies:
      - None (surface layer foundation)
    output_deliverables:
      - DockerConfig struct
      - Dockerfile generation
      - Multi-stage build support
    acceptance_criteria:
      - "DockerConfig generates valid Dockerfile"
      - "Multi-stage build separates builder and runtime"
      - "Non-root user execution configured"
      - "Health check configured for port 3000"
      - "Image size minimized with slim base"
    test_file: "crates/context-graph-testing/tests/docker_config_tests.rs"

  - id: "TASK-14-026"
    title: "Implement KubernetesConfig and Manifest Generation"
    description: |
      Implement Kubernetes configuration:

      UpdateStrategy enum:
      - RollingUpdate { max_unavailable: String, max_surge: String }
      - Recreate

      ProbeType enum:
      - HttpGet { path: String, port: u32 }
      - TcpSocket { port: u32 }
      - Exec { command: Vec<String> }
      - Grpc { port: u32 }

      Probe struct:
      - probe_type: ProbeType
      - initial_delay_seconds: u32
      - period_seconds: u32
      - timeout_seconds: u32
      - success_threshold: u32
      - failure_threshold: u32

      ResourceLimits struct:
      - cpu_request: String
      - cpu_limit: String
      - memory_request: String
      - memory_limit: String
      - gpu_request: Option<String>

      ContainerSpec struct:
      - name: String
      - image: String
      - ports: Vec<ContainerPort>
      - resources: ResourceLimits
      - liveness_probe: Option<Probe>
      - readiness_probe: Option<Probe>
      - startup_probe: Option<Probe>

      DeploymentConfig struct:
      - replicas: u32 (default: 3)
      - strategy: UpdateStrategy
      - pod_template: PodTemplate

      HPAConfig struct:
      - min_replicas: u32
      - max_replicas: u32
      - metrics: Vec<ScalingMetric>

      KubernetesConfig struct:
      - namespace: String
      - deployment: DeploymentConfig
      - service: ServiceConfig
      - config_maps: Vec<ConfigMapDef>
      - secrets: Vec<SecretRef>
      - hpa: Option<HPAConfig>
      - pdb: Option<PDBConfig>

      Methods:
      - generate_manifests(&self) -> Vec<(String, String)>
      - validate(&self) -> Result<(), ConfigError>
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/production/kubernetes.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-006"
    input_dependencies:
      - None (surface layer foundation)
    output_deliverables:
      - KubernetesConfig struct
      - Deployment, Service, HPA manifests
      - YAML generation
    acceptance_criteria:
      - "KubernetesConfig generates valid YAML manifests"
      - "Deployment has 3 replicas by default"
      - "Probes configured for liveness, readiness, startup"
      - "HPA configured for CPU/memory autoscaling"
      - "kubectl dry-run validates manifests"
    test_file: "crates/context-graph-testing/tests/kubernetes_config_tests.rs"

  - id: "TASK-14-027"
    title: "Implement ProductionConfig with Environment Management"
    description: |
      Implement production configuration management:

      EnvironmentConfig struct:
      - name: String
      - replicas: u32
      - resources: ResourceLimits
      - feature_flags: HashMap<String, bool>
      - secrets_source: SecretsSource

      SecretsSource enum:
      - Vault { path: String }
      - Kubernetes { secret_name: String }
      - EnvironmentVariables

      HealthCheckConfig struct:
      - liveness_path: String
      - readiness_path: String
      - startup_path: String
      - port: u16

      ScalingConfig struct:
      - min_replicas: u32
      - max_replicas: u32
      - cpu_target: u32
      - memory_target: u32
      - custom_metrics: Vec<CustomMetric>

      ProductionConfig struct:
      - docker: DockerConfig
      - kubernetes: KubernetesConfig
      - environments: HashMap<String, EnvironmentConfig>
      - resources: ResourceRequirements
      - health: HealthCheckConfig
      - scaling: ScalingConfig

      Methods:
      - new() -> Self
      - for_environment(&self, env: &str) -> Option<&EnvironmentConfig>
      - generate_all_configs(&self, env: &str) -> Result<ConfigBundle>
      - validate_all(&self) -> Result<(), Vec<ConfigError>>
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/production/config.rs"
    dependencies:
      - "TASK-14-025"
      - "TASK-14-026"
    requirements_traced:
      - "REQ-TEST-006"
    input_dependencies:
      - DockerConfig from TASK-14-025
      - KubernetesConfig from TASK-14-026
    output_deliverables:
      - ProductionConfig struct
      - EnvironmentConfig struct
      - ConfigBundle generation
    acceptance_criteria:
      - "ProductionConfig combines Docker and Kubernetes configs"
      - "Environment-specific overrides supported"
      - "Secrets management via Vault or K8s secrets"
      - "Health check configuration unified"
      - "Scaling parameters configurable per environment"
    test_file: "crates/context-graph-testing/tests/production_config_tests.rs"

  # ============================================================
  # SURFACE LAYER: Monitoring Infrastructure
  # ============================================================

  - id: "TASK-14-028"
    title: "Implement PrometheusConfig and MetricDefinitions"
    description: |
      Implement Prometheus metrics configuration:

      MetricType enum:
      - Counter
      - Gauge
      - Histogram
      - Summary

      MetricDefinition struct:
      - name: String
      - metric_type: MetricType
      - help: String
      - labels: Vec<String>
      - buckets: Option<Vec<f64>>

      PrometheusConfig struct:
      - endpoint: String (default: "/metrics")
      - port: u16 (default: 9090)
      - prefix: String (default: "context_graph_")
      - default_labels: HashMap<String, String>
      - latency_buckets: Vec<f64>
      - metrics: Vec<MetricDefinition>

      Required metrics (17 total):
      - context_graph_requests_total (Counter)
      - context_graph_request_duration_seconds (Histogram)
      - context_graph_active_connections (Gauge)
      - context_graph_contexts_created_total (Counter)
      - context_graph_contexts_queried_total (Counter)
      - context_graph_context_storage_bytes (Gauge)
      - context_graph_embedding_duration_seconds (Histogram)
      - context_graph_embedding_batch_size (Histogram)
      - context_graph_hnsw_search_duration_seconds (Histogram)
      - context_graph_hnsw_index_size (Gauge)
      - context_graph_graph_traversal_duration_seconds (Histogram)
      - context_graph_memory_bytes (Gauge)
      - context_graph_cpu_seconds_total (Counter)
      - context_graph_goroutines (Gauge)
      - context_graph_errors_total (Counter)
      - context_graph_recovery_time_seconds (Histogram)
      - context_graph_health_status (Gauge)

      Methods:
      - generate_metrics(&self) -> Vec<Box<dyn Metric>>
      - register_all(&self, registry: &Registry)
    layer: "surface"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/monitoring/prometheus.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-007"
    input_dependencies:
      - None (surface layer foundation)
    output_deliverables:
      - PrometheusConfig struct
      - MetricDefinition struct
      - 17 standard metrics
    acceptance_criteria:
      - "PrometheusConfig exposes /metrics endpoint"
      - "All 17 required metrics defined"
      - "Latency buckets configured for sub-second precision"
      - "Labels support request type, module, status"
      - "Prometheus library integration"
    test_file: "crates/context-graph-testing/tests/prometheus_tests.rs"

  - id: "TASK-14-029"
    title: "Implement AlertRule and AlertingConfig"
    description: |
      Implement alerting configuration:

      AlertSeverity enum:
      - Info
      - Warning
      - Critical
      - Page

      AlertRule struct:
      - name: String
      - expr: String (PromQL)
      - for_duration: Duration
      - severity: AlertSeverity
      - labels: HashMap<String, String>
      - annotations: HashMap<String, String>

      Required alerts (9 total):
      - ContextGraphDown (Critical): up == 0 for 1m
      - HighErrorRate (Warning): rate(errors_total[5m]) > 0.01 for 2m
      - VeryHighErrorRate (Critical): rate(errors_total[5m]) > 0.05 for 1m
      - HighLatency (Warning): p95 > 100ms for 5m
      - VeryHighLatency (Critical): p99 > 500ms for 2m
      - HighMemoryUsage (Warning): memory > 80% for 5m
      - HighCPUUsage (Warning): cpu > 90% for 5m
      - HighQueueDepth (Warning): queue_depth > 1000 for 5m
      - NearCapacity (Critical): active_connections > 9000 for 2m

      AlertingConfig struct:
      - rules: Vec<AlertRule>
      - evaluation_interval: Duration
      - notification_channels: Vec<NotificationChannel>

      Methods:
      - generate_prometheus_rules(&self) -> String (YAML)
      - validate_promql(&self) -> Result<(), Vec<PromqlError>>
    layer: "surface"
    priority: "high"
    estimated_hours: 2.5
    file_path: "crates/context-graph-testing/src/monitoring/alerts.rs"
    dependencies:
      - "TASK-14-028"
    requirements_traced:
      - "REQ-TEST-007"
    input_dependencies:
      - PrometheusConfig from TASK-14-028
    output_deliverables:
      - AlertRule struct
      - AlertingConfig struct
      - 9 standard alert rules
    acceptance_criteria:
      - "9 required alert rules defined"
      - "PromQL expressions are valid"
      - "Severity levels appropriate for each alert"
      - "YAML output compatible with Prometheus Alertmanager"
      - "promtool check rules passes"
    test_file: "crates/context-graph-testing/tests/alerts_tests.rs"

  - id: "TASK-14-030"
    title: "Implement MonitoringStack Configuration"
    description: |
      Implement complete monitoring stack:

      LoggingConfig struct:
      - format: LogFormat (Json, Text)
      - level: LogLevel
      - output: LogOutput (Stdout, File, Both)
      - structured_fields: Vec<String>

      TracingConfig struct:
      - enabled: bool
      - sampling_rate: f64
      - exporter: TracingExporter (Jaeger, Zipkin, OTLP)
      - service_name: String

      Dashboard struct:
      - name: String
      - panels: Vec<DashboardPanel>
      - variables: Vec<TemplateVariable>

      MonitoringConfig struct:
      - prometheus: PrometheusConfig
      - logging: LoggingConfig
      - tracing: TracingConfig
      - alerts: Vec<AlertRule>
      - dashboards: Vec<Dashboard>

      Methods:
      - new() -> Self
      - generate_grafana_dashboards(&self) -> Vec<(String, String)>
      - generate_logging_config(&self) -> String
      - generate_tracing_config(&self) -> String
      - validate_all(&self) -> Result<(), Vec<MonitoringError>>
    layer: "surface"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/monitoring/stack.rs"
    dependencies:
      - "TASK-14-028"
      - "TASK-14-029"
    requirements_traced:
      - "REQ-TEST-007"
    input_dependencies:
      - PrometheusConfig from TASK-14-028
      - AlertRule from TASK-14-029
    output_deliverables:
      - MonitoringConfig struct
      - Grafana dashboard generation
      - Logging and tracing configs
    acceptance_criteria:
      - "MonitoringConfig unifies Prometheus, logging, tracing"
      - "Grafana dashboard JSON generation"
      - "Structured JSON logging configured"
      - "Distributed tracing with sampling"
      - "All configs validate successfully"
    test_file: "crates/context-graph-testing/tests/monitoring_stack_tests.rs"

  # ============================================================
  # SURFACE LAYER: Security Testing
  # ============================================================

  - id: "TASK-14-031"
    title: "Implement SecurityTestConfig and FuzzingConfig"
    description: |
      Implement security testing configuration:

      DependencyAuditConfig struct:
      - tool: String (cargo-audit)
      - advisory_db: String
      - fail_threshold: SecuritySeverity
      - ignore: Vec<String>
      - report_path: PathBuf

      SecuritySeverity enum:
      - None
      - Low
      - Medium
      - High
      - Critical

      FuzzingTool enum:
      - CargoFuzz
      - AFL
      - LibFuzzer

      FuzzInputType enum:
      - Bytes
      - Json
      - Protobuf
      - Custom(String)

      FuzzTarget struct:
      - name: String
      - function: String
      - input_type: FuzzInputType
      - max_size: usize
      - seeds: Vec<Vec<u8>>

      Required fuzz targets:
      - fuzz_json_rpc_parse
      - fuzz_mcp_request_validate
      - fuzz_memory_node_deserialize
      - fuzz_embedding_input
      - fuzz_graph_query_parse
      - fuzz_utl_decompress

      FuzzingConfig struct:
      - tool: FuzzingTool
      - targets: Vec<FuzzTarget>
      - max_duration: Duration
      - corpus_dir: PathBuf
      - artifact_dir: PathBuf

      SecurityTestConfig struct:
      - dependency_audit: DependencyAuditConfig
      - fuzzing: FuzzingConfig
      - static_analysis: StaticAnalysisConfig
      - penetration: PenetrationTestConfig
    layer: "surface"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-testing/src/security/config.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-008"
    input_dependencies:
      - None (surface layer foundation)
    output_deliverables:
      - SecurityTestConfig struct
      - FuzzingConfig struct
      - 6 fuzz targets defined
    acceptance_criteria:
      - "DependencyAuditConfig uses cargo-audit"
      - "6 required fuzz targets defined"
      - "SecuritySeverity threshold configurable"
      - "Fuzzing corpus and artifacts managed"
      - "Static analysis via clippy/semgrep"
    test_file: "crates/context-graph-testing/tests/security_config_tests.rs"

  # ============================================================
  # SURFACE LAYER: Marblestone Test Cases
  # ============================================================

  - id: "TASK-14-032"
    title: "Implement Marblestone Unit Tests (T-MARB-001 to T-MARB-008)"
    description: |
      Implement 8 Marblestone-specific unit test cases:

      T-MARB-001: Neurotransmitter Edge Weights
      - Create edge with NeurotransmitterWeights
      - get_modulated_weight() with high excitatory
      - get_modulated_weight() with high inhibitory
      - domain_aware_search() by domain

      T-MARB-002: Lifecycle Lambda Weights
      - Infancy stage lambda weights (DS=0.7, DC=0.3)
      - Growth stage lambda weights (DS=0.5, DC=0.5)
      - Maturity stage lambda weights (DS=0.3, DC=0.7)
      - Stage transition updates lambdas

      T-MARB-003: Formal Verification Layer
      - Verify valid code node returns Verified
      - Verify code with bounds error returns Failed
      - Verify non-code content returns NotApplicable
      - Verification timeout after configured ms

      T-MARB-004: Amortized Shortcut Creation
      - Create shortcut from 3-hop path
      - Reject 2-hop path
      - Reject low confidence path (<0.7)
      - Shortcut weight = product of path weights

      T-MARB-005: Steering Dopamine Feedback
      - Positive steering reward increases dopamine
      - Negative steering reward decreases dopamine
      - Dopamine clamping in [0, 1]
      - Steering source tracking

      T-MARB-006: Omnidirectional Inference
      - Forward inference A->B
      - Backward inference B->A
      - Clamped variable inference
      - Abductive inference

      T-MARB-007: Steering Subsystem
      - SteeringReward returned on store
      - Gardener evaluation
      - Curator evaluation
      - Thought Assessor evaluation

      T-MARB-008: New MCP Tools
      - get_steering_feedback MCP call
      - omni_infer MCP call
      - verify_code_node MCP call
    layer: "surface"
    priority: "high"
    estimated_hours: 6
    file_path: "crates/context-graph-testing/tests/marblestone/unit_tests.rs"
    dependencies: []
    requirements_traced:
      - "REQ-TEST-011"
    input_dependencies:
      - Module 2, 4, 6, 9, 10, 11, 12, 13 implementations
    output_deliverables:
      - 8 test case implementations
      - 31 individual test functions
    acceptance_criteria:
      - "All 8 T-MARB test cases implemented"
      - "Each test case has 4 test functions"
      - "Tests cover all Marblestone features"
      - "Tests use real implementations, no mocks"
      - "cargo nextest run --test marblestone passes"
    test_file: "crates/context-graph-testing/tests/marblestone/unit_tests.rs"

  - id: "TASK-14-033"
    title: "Implement Marblestone Integration Tests (MI-001 to MI-006)"
    description: |
      Implement 6 Marblestone integration tests:

      MI-001: Marblestone-HNSW Integration
      - Generate embeddings through Marblestone
      - Index in HNSW
      - Query through Marblestone encoding
      - Verify search results

      MI-002: Marblestone-Temporal Integration
      - Create time-aware contexts
      - Encode with Marblestone
      - Query temporal range
      - Verify 40 results for 50-10 hour range

      MI-003: Marblestone-Memory Consolidation
      - Store 500 short-term memories
      - Trigger consolidation
      - Verify long-term memory count < 500
      - Verify retrieval still works

      MI-004: Marblestone-Query Pipeline
      - Enable Marblestone encoding
      - Insert 1000 contexts
      - Execute complex semantic+temporal query
      - Verify relevance scores > 0.5

      MI-005: Marblestone-Federation Sync
      - Insert on node A
      - Sync to node B
      - Query on node B finds contexts
      - Verify embedding consistency (cosine > 0.99)

      MI-006: Marblestone-Security Integration
      - Create contexts with security levels
      - Query as public user
      - Verify only public contexts returned
    layer: "surface"
    priority: "high"
    estimated_hours: 5
    file_path: "crates/context-graph-testing/tests/marblestone/integration_tests.rs"
    dependencies:
      - "TASK-14-015"
    requirements_traced:
      - "REQ-TEST-011"
    input_dependencies:
      - IntegrationTestHarness from TASK-14-015
      - All module implementations
    output_deliverables:
      - 6 integration test implementations
      - Cross-module validation
    acceptance_criteria:
      - "All 6 MI integration tests implemented"
      - "Tests validate cross-module interactions"
      - "Tests use real module instances"
      - ">80% cross-module coverage"
      - "cargo nextest run --test marblestone_integration passes"
    test_file: "crates/context-graph-testing/tests/marblestone/integration_tests.rs"

  - id: "TASK-14-034"
    title: "Implement Marblestone Benchmarks (B-MARB-001 to B-MARB-007)"
    description: |
      Implement 7 Marblestone performance benchmarks:

      B-MARB-001: Forward Pass Throughput
      - Batch sizes: 1, 8, 32, 64, 128
      - Target: <8ms for batch=32

      B-MARB-002: Routing Latency
      - Target: <1ms

      B-MARB-003: Memory Access Patterns
      - Read: <0.1ms
      - Write: <0.2ms
      - Search: <3ms

      B-MARB-004: Attention Computation
      - Sequence lengths: 64, 128, 256, 512
      - Target: <12ms for seq=256

      B-MARB-005: Consolidation Throughput
      - Memory counts: 100, 500, 1000, 5000
      - Target: <60ms for n=1000

      B-MARB-006: Module Switching Overhead
      - Target: <0.5ms

      B-MARB-007: End-to-End Encoding
      - Text lengths: 100, 500, 1000, 2000 chars
      - Target: <15ms for 1000 chars

      Use criterion with statistical analysis.
      Variance must be <2%.
    layer: "surface"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/benches/marblestone_benchmarks.rs"
    dependencies:
      - "TASK-14-018"
    requirements_traced:
      - "REQ-TEST-011"
    input_dependencies:
      - PerformanceBenchmarks from TASK-14-018
    output_deliverables:
      - 7 benchmark implementations
      - Criterion integration
      - Baseline measurements
    acceptance_criteria:
      - "All 7 B-MARB benchmarks implemented"
      - "Criterion statistical analysis used"
      - "Variance <2% across runs"
      - "Baselines stored for regression detection"
      - "cargo bench --bench marblestone passes"
    test_file: "crates/context-graph-testing/benches/marblestone_benchmarks.rs"

  # ============================================================
  # SURFACE LAYER: CI/CD Pipeline
  # ============================================================

  - id: "TASK-14-035"
    title: "Implement CI/CD Pipeline Configuration"
    description: |
      Implement CI/CD pipeline configuration:

      CIPipelineStage struct:
      - name: String
      - duration_target: Duration
      - dependencies: Vec<String>
      - actions: Vec<CIAction>

      CIAction enum:
      - Check
      - Fmt
      - Clippy
      - Test { filter: Option<String> }
      - Coverage { threshold: f64 }
      - Audit
      - Benchmark
      - Build { release: bool }
      - DockerBuild
      - DockerScan

      Pipeline stages (8 total):
      1. quick-checks: <2min (fmt, clippy, check)
      2. unit-tests: <5min (cargo nextest --lib)
      3. coverage: <10min (llvm-cov, fail-under 90%)
      4. integration-tests: <15min (nextest integration)
      5. security: <3min (cargo-audit)
      6. benchmarks: weekly (criterion)
      7. build-release: <5min (cargo build --release)
      8. docker: <10min (build + trivy scan)

      CIPipelineConfig struct:
      - stages: Vec<CIPipelineStage>
      - parallel_stages: Vec<Vec<String>>
      - total_timeout: Duration (target: <30min)

      Methods:
      - generate_github_actions(&self) -> String
      - generate_gitlab_ci(&self) -> String
      - validate(&self) -> Result<(), PipelineError>

      Constraint: Total pipeline <30 minutes.
    layer: "surface"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-testing/src/ci/pipeline.rs"
    dependencies:
      - "TASK-14-010"
      - "TASK-14-015"
      - "TASK-14-018"
      - "TASK-14-031"
    requirements_traced:
      - "REQ-TEST-010"
    input_dependencies:
      - UnitTestFramework from TASK-14-010
      - IntegrationTestHarness from TASK-14-015
      - PerformanceBenchmarks from TASK-14-018
      - SecurityTestConfig from TASK-14-031
    output_deliverables:
      - CIPipelineConfig struct
      - GitHub Actions workflow
      - GitLab CI configuration
    acceptance_criteria:
      - "8 pipeline stages defined"
      - "Parallel execution where possible"
      - "Total pipeline <30 minutes"
      - "Coverage threshold enforced at 90%"
      - "Security scan blocks on vulnerabilities"
      - "GitHub Actions YAML generation"
    test_file: "crates/context-graph-testing/tests/pipeline_tests.rs"
```

---

## Dependency Graph

```
TASK-14-001 (TestDiscoveryConfig) ─────────────────────────────────────────────┐
TASK-14-002 (CoverageConfig) ──────────────────────────────────────────────────┤
TASK-14-003 (ExecutionConfig) ─────────────────────────────────────────────────┤
TASK-14-004 (MockRegistry) ────────────────────────────────────────────────────┤
TASK-14-005 (TestFixture, FixtureManager) ─────────────────────────────────────┤
                                                                                │
TASK-14-006 (FlakyTestDetector) ───────────────────────────────────────────────┤
                                                                                │
TASK-14-006 ──► TASK-14-007 (TestResults) ─────────────────────────────────────┤
TASK-14-002 ──► TASK-14-008 (CoverageReport) ──────────────────────────────────┤
TASK-14-007 ──► TASK-14-009 (TestResultAggregator) ────────────────────────────┤
                                                                                │
ALL ABOVE ──► TASK-14-010 (UnitTestFramework) ─────────────────────────────────┤
                                                                                │
TASK-14-011 (IsolationManager) ────────────────────────────────────────────────┤
TASK-14-012 (ModuleOrchestrator) ──────────────────────────────────────────────┤
TASK-14-012 ──► TASK-14-013 (IntegrationTestSuite) ────────────────────────────┤
TASK-14-013 ──► TASK-14-014 (IntegrationResultCollector) ──────────────────────┤
                                                                                │
TASK-14-005 + TASK-14-011 + TASK-14-012 + TASK-14-013 + TASK-14-014            │
                              ──► TASK-14-015 (IntegrationTestHarness) ─────────┤
                                                                                │
TASK-14-012 ──► TASK-14-016 (BenchmarkDefinition) ─────────────────────────────┤
TASK-14-016 ──► TASK-14-017 (BaselineRegistry) ────────────────────────────────┤
TASK-14-016 + TASK-14-017 ──► TASK-14-018 (PerformanceBenchmarks) ─────────────┤
                                                                                │
TASK-14-019 (LoadProfile) ─────────────────────────────────────────────────────┤
TASK-14-019 ──► TASK-14-020 (LoadMetricsCollector) ────────────────────────────┤
TASK-14-019 + TASK-14-020 ──► TASK-14-021 (LoadTester) ────────────────────────┤
                                                                                │
TASK-14-022 (FaultType, FaultRegistry) ────────────────────────────────────────┤
TASK-14-022 ──► TASK-14-023 (ChaosExperiment, SafetyControls) ─────────────────┤
TASK-14-022 + TASK-14-023 ──► TASK-14-024 (ChaosEngine) ───────────────────────┤
                                                                                │
TASK-14-025 (DockerConfig) ────────────────────────────────────────────────────┤
TASK-14-026 (KubernetesConfig) ────────────────────────────────────────────────┤
TASK-14-025 + TASK-14-026 ──► TASK-14-027 (ProductionConfig) ──────────────────┤
                                                                                │
TASK-14-028 (PrometheusConfig) ────────────────────────────────────────────────┤
TASK-14-028 ──► TASK-14-029 (AlertingConfig) ──────────────────────────────────┤
TASK-14-028 + TASK-14-029 ──► TASK-14-030 (MonitoringStack) ───────────────────┤
                                                                                │
TASK-14-031 (SecurityTestConfig) ──────────────────────────────────────────────┤
                                                                                │
TASK-14-032 (Marblestone Unit Tests) ──────────────────────────────────────────┤
TASK-14-015 ──► TASK-14-033 (Marblestone Integration Tests) ───────────────────┤
TASK-14-018 ──► TASK-14-034 (Marblestone Benchmarks) ──────────────────────────┤
                                                                                │
ALL ABOVE ──► TASK-14-035 (CI/CD Pipeline) ◄───────────────────────────────────┘
```

---

## Implementation Order (Recommended)

### Week 1: Foundation + Unit Testing
1. TASK-14-001: TestDiscoveryConfig
2. TASK-14-002: CoverageConfig
3. TASK-14-003: ExecutionConfig
4. TASK-14-004: MockRegistry
5. TASK-14-005: TestFixture, FixtureManager
6. TASK-14-006: FlakyTestDetector
7. TASK-14-007: TestResults
8. TASK-14-008: CoverageReport
9. TASK-14-009: TestResultAggregator
10. TASK-14-010: UnitTestFramework

### Week 2: Integration + Benchmarks
11. TASK-14-011: IsolationManager
12. TASK-14-012: ModuleOrchestrator
13. TASK-14-013: IntegrationTestSuite
14. TASK-14-014: IntegrationResultCollector
15. TASK-14-015: IntegrationTestHarness
16. TASK-14-016: BenchmarkDefinition
17. TASK-14-017: BaselineRegistry
18. TASK-14-018: PerformanceBenchmarks

### Week 3: Load + Chaos + Production
19. TASK-14-019: LoadProfile
20. TASK-14-020: LoadMetricsCollector
21. TASK-14-021: LoadTester
22. TASK-14-022: FaultType, FaultRegistry
23. TASK-14-023: ChaosExperiment, SafetyControls
24. TASK-14-024: ChaosEngine
25. TASK-14-025: DockerConfig
26. TASK-14-026: KubernetesConfig
27. TASK-14-027: ProductionConfig

### Week 4: Monitoring + Security + Marblestone + CI/CD
28. TASK-14-028: PrometheusConfig
29. TASK-14-029: AlertingConfig
30. TASK-14-030: MonitoringStack
31. TASK-14-031: SecurityTestConfig
32. TASK-14-032: Marblestone Unit Tests
33. TASK-14-033: Marblestone Integration Tests
34. TASK-14-034: Marblestone Benchmarks
35. TASK-14-035: CI/CD Pipeline

---

## Quality Gates

| Gate | Criteria | Required For |
|------|----------|--------------|
| Foundation Complete | TASK-14-001 through TASK-14-010 pass all tests | Week 2 start |
| Integration Complete | TASK-14-015 passes all tests | Week 3 start |
| Benchmarks Complete | TASK-14-018 shows <2% variance | Week 3 start |
| Load/Chaos Complete | TASK-14-021, TASK-14-024 pass | Week 4 start |
| Production Ready | TASK-14-027 generates valid configs | Module complete |
| Marblestone Tests Pass | All T-MARB, MI, B-MARB tests pass | Module complete |
| CI Pipeline Ready | TASK-14-035 generates valid workflows | Final gate |

---

## Performance Targets Summary

| Component | Budget | Metric |
|-----------|--------|--------|
| Unit Test Suite | <5 minutes | Full workspace execution |
| Integration Tests | <15 minutes | Cross-module scenarios |
| Benchmark Variance | <2% | Criterion statistical analysis |
| Load Test Capacity | 10,000 | Concurrent sessions |
| MTTR | <30 seconds | Chaos fault recovery |
| Container Startup | <10 seconds | Production deployment |
| CI Pipeline | <30 minutes | All stages parallel |
| Coverage Threshold | >90% | Line coverage |
| Integration Coverage | >80% | Cross-module paths |

---

## Memory Budget

| Component | Budget |
|-----------|--------|
| Test Results Buffer | <100MB |
| Coverage Data | <500MB |
| HDR Histogram (load test) | <50MB |
| Chaos State Tracking | <10MB |
| Benchmark Baselines | <10MB |

---

## Traceability Matrix

| Task ID | Requirements Covered |
|---------|---------------------|
| TASK-14-001 | REQ-TEST-001 |
| TASK-14-002 | REQ-TEST-001 |
| TASK-14-003 | REQ-TEST-001 |
| TASK-14-004 | REQ-TEST-001 |
| TASK-14-005 | REQ-TEST-001, REQ-TEST-002 |
| TASK-14-006 | REQ-TEST-001 |
| TASK-14-007 | REQ-TEST-001 |
| TASK-14-008 | REQ-TEST-001 |
| TASK-14-009 | REQ-TEST-001 |
| TASK-14-010 | REQ-TEST-001 |
| TASK-14-011 | REQ-TEST-002 |
| TASK-14-012 | REQ-TEST-002 |
| TASK-14-013 | REQ-TEST-002 |
| TASK-14-014 | REQ-TEST-002 |
| TASK-14-015 | REQ-TEST-002 |
| TASK-14-016 | REQ-TEST-003 |
| TASK-14-017 | REQ-TEST-003 |
| TASK-14-018 | REQ-TEST-003 |
| TASK-14-019 | REQ-TEST-004 |
| TASK-14-020 | REQ-TEST-004 |
| TASK-14-021 | REQ-TEST-004 |
| TASK-14-022 | REQ-TEST-005 |
| TASK-14-023 | REQ-TEST-005 |
| TASK-14-024 | REQ-TEST-005 |
| TASK-14-025 | REQ-TEST-006 |
| TASK-14-026 | REQ-TEST-006 |
| TASK-14-027 | REQ-TEST-006 |
| TASK-14-028 | REQ-TEST-007 |
| TASK-14-029 | REQ-TEST-007 |
| TASK-14-030 | REQ-TEST-007 |
| TASK-14-031 | REQ-TEST-008 |
| TASK-14-032 | REQ-TEST-011 |
| TASK-14-033 | REQ-TEST-011 |
| TASK-14-034 | REQ-TEST-011 |
| TASK-14-035 | REQ-TEST-010 |

---

## Critical Constraints

**COVERAGE THRESHOLDS ARE HARD REQUIREMENTS**:
- Unit Test Coverage: >90% line coverage
- Branch Coverage: >85%
- Function Coverage: >95%
- Integration Coverage: >80% cross-module paths

**PERFORMANCE BUDGETS ARE HARD REQUIREMENTS**:
- Unit Tests: <5 minutes total
- Integration Tests: <15 minutes total
- Benchmark Variance: <2%
- Load Capacity: 10,000 concurrent sessions
- MTTR: <30 seconds
- CI Pipeline: <30 minutes

**PRODUCTION READINESS**:
- Container startup: <10 seconds
- Health check response: <30 seconds
- Zero critical security vulnerabilities
- All Marblestone tests passing

---

*Generated: 2025-12-31*
*Module: 14 - Testing & Production Infrastructure*
*Version: 1.0.0*
*Total Tasks: 35*
