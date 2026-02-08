//! Chaos Testing Suite for Context Graph Resilience Validation
//!
//! TASK-TEST-P2-001: Chaos engineering tests to validate system resilience.
//!
//! Constitution Reference: testing.types.chaos
//! - GPU OOM recovery
//! - Concurrent mutation (100 writers)
//! - Memory pressure cycling
//! - Resource exhaustion recovery
//!
//! # Running Chaos Tests
//!
//! ```bash
//! # Run all chaos tests (included in regular test suite)
//! cargo test --test chaos_tests -- --nocapture
//!
//! # Run a specific test
//! cargo test --test chaos_tests test_gpu_oom_detection -- --nocapture
//! ```
//!
//! # Test Categories (16 tests total)
//!
//! ## GPU OOM Recovery (4 tests)
//! - `test_gpu_oom_detection` - OOM returns Err, not panic
//! - `test_gpu_oom_recovery_with_deallocation` - Recovery after free
//! - `test_gpu_cascading_oom_prevention` - Category isolation
//! - `test_gpu_oom_error_propagation` - Error context quality
//!
//! ## Concurrent Mutation Stress (4 tests)
//! - `test_100_concurrent_writers` - 100 threads with barrier sync
//! - `test_concurrent_read_write_mix` - 50 writers + 50 readers
//! - `test_concurrent_allocation_contention` - Limited budget contention
//! - `test_thundering_herd_scenario` - Mass release and grab
//!
//! ## Memory Pressure (4 tests)
//! - `test_allocation_deallocation_cycling` - 10000 cycles, no leak
//! - `test_fragmentation_under_pressure` - Alternating alloc/free
//! - `test_low_memory_threshold_warning` - 90% threshold detection
//! - `test_memory_usage_accounting_accuracy` - Invariant: used + available == budget
//!
//! ## Resource Exhaustion (4 tests)
//! - `test_graceful_degradation_on_exhaustion` - System stays responsive
//! - `test_recovery_after_resource_release` - Full capacity restored
//! - `test_category_budget_exhaustion` - Category isolation
//! - `test_allocation_rejection_messages` - Informative error messages
//!
//! # Constitution Compliance
//!
//! - AP-001: Fail fast - all tests verify Err returned, not panic
//! - perf.memory.gpu: <24GB - tests use GpuMemoryManager for budget tracking
//! - NO MOCKS: All tests use real GpuMemoryManager implementation

#[path = "chaos_tests/gpu_oom_recovery.rs"]
mod gpu_oom_recovery;

#[path = "chaos_tests/concurrent_mutation_stress.rs"]
mod concurrent_mutation_stress;

#[path = "chaos_tests/memory_pressure.rs"]
mod memory_pressure;

#[path = "chaos_tests/resource_exhaustion.rs"]
mod resource_exhaustion;
