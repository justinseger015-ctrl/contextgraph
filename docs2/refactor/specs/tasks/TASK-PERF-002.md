# TASK-PERF-002: Profiling Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-PERF-002 |
| **Title** | Profiling Infrastructure |
| **Status** | :white_circle: todo |
| **Layer** | Performance |
| **Sequence** | 52 |
| **Estimated Days** | 1.5 |
| **Complexity** | Medium |

## Implements

- Performance debugging capability
- Bottleneck identification
- Memory leak detection
- CPU hotspot analysis

## Dependencies

| Task | Reason |
|------|--------|
| TASK-PERF-001 | Benchmarks to profile |
| TASK-CORE-011 | GPU memory to profile |

## Objective

Set up profiling infrastructure for:
1. CPU profiling (flamegraphs)
2. Memory profiling (allocations, leaks)
3. GPU profiling (CUDA metrics)
4. Async runtime profiling (tokio-console)

## Context

Profiling complements benchmarking by showing *where* time is spent, not just *how much* time. Essential for optimization work.

## Scope

### In Scope

- Flamegraph generation
- Memory allocation tracking
- GPU utilization metrics
- Tokio runtime tracing
- Profiling documentation

### Out of Scope

- Automated optimization
- Distributed profiling
- Real-time production profiling

## Definition of Done

### CPU Profiling Setup

```rust
// Cargo.toml additions
[profile.profiling]
inherits = "release"
debug = true

[dev-dependencies]
pprof = { version = "0.13", features = ["flamegraph", "criterion"] }
```

```rust
// benches/profiling.rs

use criterion::{criterion_group, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn profiled_benchmarks(c: &mut Criterion) {
    // Configure pprof profiler
    let profiler = PProfProfiler::new(100, Output::Flamegraph(None));

    let mut group = c.benchmark_group("profiled");

    // Benchmarks with profiling enabled
    group.bench_function("search_profiled", |b| {
        b.iter(|| {
            // Search operation
        })
    });

    group.finish();
}

criterion_group! {
    name = profiled_benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = profiled_benchmarks
}
```

### Flamegraph Generation Script

```bash
#!/bin/bash
# scripts/flamegraph.sh

set -e

# Install dependencies if needed
if ! command -v perf &> /dev/null; then
    echo "Installing perf..."
    sudo apt-get install linux-tools-common linux-tools-generic
fi

# Build with profiling symbols
cargo build --profile profiling --bin context-graph-mcp

# Run with perf
sudo perf record -g --call-graph dwarf \
    ./target/profiling/context-graph-mcp --profile-mode &
PID=$!

# Run test workload
sleep 2
cargo run --example benchmark_workload
kill $PID

# Generate flamegraph
sudo perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg

echo "Flamegraph saved to flamegraph.svg"
```

### Memory Profiling

```rust
// src/profiling/memory.rs

#[cfg(feature = "memory-profiling")]
use dhat::{Dhat, DhatAlloc};

#[cfg(feature = "memory-profiling")]
#[global_allocator]
static ALLOC: DhatAlloc = DhatAlloc;

/// Memory profiling guard
pub struct MemoryProfiler {
    #[cfg(feature = "memory-profiling")]
    _dhat: Dhat,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        #[cfg(feature = "memory-profiling")]
        let _dhat = Dhat::start_heap_profiling();

        Self {
            #[cfg(feature = "memory-profiling")]
            _dhat,
        }
    }
}

impl Drop for MemoryProfiler {
    fn drop(&mut self) {
        // dhat automatically saves profile on drop
    }
}
```

```toml
# Cargo.toml
[features]
memory-profiling = ["dhat"]

[dev-dependencies]
dhat = "0.3"
```

### Memory Profiling Script

```bash
#!/bin/bash
# scripts/memory_profile.sh

set -e

# Build with memory profiling
cargo build --release --features memory-profiling

# Run workload
DHAT_SAVE_PROFILE=1 ./target/release/context-graph-mcp --test-workload

# Analyze with dhat-viewer
# dhat-viewer dhat-heap.json

echo "Memory profile saved to dhat-heap.json"
echo "Open with: dhat-viewer dhat-heap.json"
```

### GPU Profiling

```rust
// src/profiling/gpu.rs

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

/// GPU metrics collector
pub struct GpuProfiler {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    start_memory: usize,
}

impl GpuProfiler {
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        let device = CudaDevice::new(0).unwrap();

        #[cfg(feature = "cuda")]
        let (free, total) = device.mem_get_info().unwrap();

        Self {
            #[cfg(feature = "cuda")]
            device,
            start_memory: total - free,
        }
    }

    /// Get current GPU memory usage
    pub fn memory_usage(&self) -> GpuMemoryStats {
        #[cfg(feature = "cuda")]
        {
            let (free, total) = self.device.mem_get_info().unwrap();
            GpuMemoryStats {
                used: total - free,
                free,
                total,
                delta: (total - free) as i64 - self.start_memory as i64,
            }
        }

        #[cfg(not(feature = "cuda"))]
        GpuMemoryStats::default()
    }

    /// Run NVIDIA profiler (nvprof/nsight)
    pub fn start_nvprof() -> NvprofGuard {
        #[cfg(feature = "cuda")]
        {
            // Start profiling
            unsafe {
                cudarc::driver::sys::cuProfilerStart();
            }
        }
        NvprofGuard
    }
}

pub struct NvprofGuard;

impl Drop for NvprofGuard {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            cudarc::driver::sys::cuProfilerStop();
        }
    }
}

#[derive(Debug, Default)]
pub struct GpuMemoryStats {
    pub used: usize,
    pub free: usize,
    pub total: usize,
    pub delta: i64,
}
```

### GPU Profiling Script

```bash
#!/bin/bash
# scripts/gpu_profile.sh

set -e

# Build with CUDA
cargo build --release --features cuda

# Run with NVIDIA Nsight Systems
nsys profile \
    --output=gpu_profile \
    --force-overwrite=true \
    ./target/release/context-graph-mcp --test-workload

echo "GPU profile saved to gpu_profile.nsys-rep"
echo "Open with: nsys-ui gpu_profile.nsys-rep"

# Alternative: nvprof (older CUDA)
# nvprof --output-profile gpu_profile.nvvp ./target/release/context-graph-mcp
```

### Tokio Runtime Profiling

```rust
// src/profiling/async.rs

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize tokio-console for async profiling
#[cfg(feature = "tokio-console")]
pub fn init_tokio_console() {
    console_subscriber::init();
}

/// Initialize tracing for async spans
pub fn init_async_tracing() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

/// Instrument async function with timing
#[macro_export]
macro_rules! profile_async {
    ($name:expr, $body:expr) => {{
        let span = tracing::info_span!($name);
        let _enter = span.enter();
        let start = std::time::Instant::now();
        let result = $body;
        tracing::info!(
            target: "profiling",
            elapsed_ms = start.elapsed().as_millis(),
            $name
        );
        result
    }};
}
```

```toml
# Cargo.toml
[features]
tokio-console = ["console-subscriber", "tokio/tracing"]

[dev-dependencies]
console-subscriber = "0.2"
```

### Profiling Script

```bash
#!/bin/bash
# scripts/profile.sh

set -e

MODE=${1:-cpu}

case $MODE in
    cpu)
        echo "Running CPU profiling..."
        ./scripts/flamegraph.sh
        ;;
    memory)
        echo "Running memory profiling..."
        ./scripts/memory_profile.sh
        ;;
    gpu)
        echo "Running GPU profiling..."
        ./scripts/gpu_profile.sh
        ;;
    async)
        echo "Running async profiling..."
        RUSTFLAGS="--cfg tokio_unstable" cargo run --features tokio-console &
        tokio-console
        ;;
    all)
        ./scripts/profile.sh cpu
        ./scripts/profile.sh memory
        ./scripts/profile.sh gpu
        ;;
    *)
        echo "Usage: $0 {cpu|memory|gpu|async|all}"
        exit 1
        ;;
esac
```

### Documentation

```markdown
# Profiling Guide

## Quick Start

```bash
# CPU profiling (flamegraph)
./scripts/profile.sh cpu

# Memory profiling
./scripts/profile.sh memory

# GPU profiling
./scripts/profile.sh gpu

# Async/tokio profiling
./scripts/profile.sh async
```

## Interpreting Flamegraphs

- Width = time spent in function
- Height = call stack depth
- Look for wide "plateaus" (hotspots)

## Common Patterns

### Memory Leaks
- Look for unbounded growth in dhat output
- Check for missing `Drop` implementations

### CPU Hotspots
- Wide bars in flamegraph
- Often in tight loops or allocation

### GPU Bottlenecks
- High "GPU idle" time = CPU-bound
- High "Memory transfer" = bandwidth-bound
- Use memory pinning for transfers
```

### Constraints

| Constraint | Target |
|------------|--------|
| Flamegraph generation | < 5 minutes |
| Memory profile overhead | < 10% |
| Profile data size | < 100MB |

## Verification

- [ ] Flamegraph generation works
- [ ] Memory profiler captures allocations
- [ ] GPU profiler captures CUDA metrics
- [ ] Tokio console connects
- [ ] Scripts documented and working

## Files to Create

| File | Purpose |
|------|---------|
| `src/profiling/mod.rs` | Profiling module |
| `src/profiling/memory.rs` | Memory profiling |
| `src/profiling/gpu.rs` | GPU profiling |
| `src/profiling/async.rs` | Async profiling |
| `scripts/flamegraph.sh` | Flamegraph script |
| `scripts/memory_profile.sh` | Memory script |
| `scripts/gpu_profile.sh` | GPU script |
| `scripts/profile.sh` | Main profile script |
| `docs/profiling.md` | Documentation |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Profiler overhead | Medium | Low | Separate profile builds |
| Platform differences | Medium | Low | Document requirements |
| Complex setup | Medium | Low | Scripts automate |

## Traceability

- Source: Performance optimization requirements
- Related: TASK-PERF-001 (Benchmarks)
