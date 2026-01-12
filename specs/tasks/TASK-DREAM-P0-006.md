# TASK-DREAM-P0-006: Wake Controller and MCP Integration

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-006 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 3 (Surface/Integration) |
| **Priority** | P0 - Critical |
| **Effort** | 4 hours |
| **Dependencies** | TASK-DREAM-P0-003 (HebbianEngine), TASK-DREAM-P0-004 (HyperbolicExplorer), TASK-DREAM-P0-005 (TriggerManager) |
| **Blocks** | None (Final task in chain) |

---

## CRITICAL: Current State Audit (2026-01-12)

### Files That ALREADY EXIST

| File | Status | Key Exports |
|------|--------|-------------|
| `dream/mod.rs` | EXISTS | `WakeReason` enum, constants module, all re-exports |
| `dream/controller.rs` | EXISTS | `DreamController`, `DreamState`, `DreamReport`, `DreamStatus` |
| `dream/triggers.rs` | EXISTS | `TriggerManager`, `GpuMonitor`, `EntropyCalculator` |
| `dream/types.rs` | EXISTS | `ExtendedTriggerReason`, `GpuTriggerState`, `EntropyWindow`, etc. |
| `dream/hebbian.rs` | EXISTS | `HebbianEngine`, `EdgeUpdate`, `HebbianUpdateResult` |
| `dream/hyperbolic_walk.rs` | EXISTS | `HyperbolicExplorer`, `DiscoveredBlindSpot`, `ExplorationResult` |

### Files That DO NOT EXIST (Must Create)

| File | Purpose |
|------|---------|
| `dream/wake_controller.rs` | Fast wake interruption with <100ms latency guarantee |
| `dream/mcp_events.rs` | MCP event definitions for GWT workspace integration |

### Key Types Already Available

```rust
// From mod.rs - WakeReason already exists
pub enum WakeReason {
    ExternalQuery,
    GpuOverBudget,
    CycleComplete,
    ManualAbort,
    ResourcePressure,
    Error,
}

// From triggers.rs - GpuMonitor already exists
pub struct GpuMonitor {
    simulated_usage: f32,
    use_simulated: bool,
}

// From types.rs - ExtendedTriggerReason already exists
pub enum ExtendedTriggerReason {
    IdleTimeout, HighEntropy, GpuOverload, MemoryPressure, Manual, Scheduled,
}
```

### Current DreamController Structure (controller.rs)

The existing `DreamController` has:
- `interrupt_flag: Arc<AtomicBool>` - Basic interrupt mechanism
- `abort()` method - Returns wake latency
- GPU budget checking via `check_gpu_budget()`
- BUT lacks dedicated `WakeController` with formal state machine

---

## 1. Objective

Implement dedicated `WakeController` for managing dream cycle interruption with:
1. Guaranteed <100ms wake latency (Constitution requirement)
2. GPU budget enforcement (<30%) during dreams
3. MCP event broadcasting for GWT workspace integration
4. Formal state machine for wake transitions

---

## 2. Constitution Reference

From `docs2/constitution.yaml`:

```yaml
dream:
  constraints:
    wake: "<100ms"        # Line 273 - MANDATORY latency guarantee
    gpu: "<30%"           # Line 273 - Budget during dream
    abort_on_query: true  # Line 273 - Immediate wake on external query

perf:
  latency:
    dream_wake: "<100ms"  # Line 130 - Performance budget
```

---

## 3. Files to Create/Modify

### 3.1 CREATE: `crates/context-graph-core/src/dream/wake_controller.rs`

```rust
//! Wake Controller - Fast Dream Interruption System
//!
//! Manages wake transitions with guaranteed <100ms latency as required by
//! Constitution Section dream.constraints.wake.
//!
//! ## Constitution Reference
//!
//! - wake: <100ms latency (docs2/constitution.yaml line 273)
//! - gpu: <30% usage during dream (docs2/constitution.yaml line 273)
//! - abort_on_query: true (docs2/constitution.yaml line 273)

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use super::constants;
use super::triggers::GpuMonitor;
use super::WakeReason;

/// Error types for wake controller operations.
#[derive(Debug, Error)]
pub enum WakeError {
    #[error("Wake latency exceeded: {actual_ms}ms > {max_ms}ms (Constitution violation)")]
    LatencyViolation { actual_ms: u64, max_ms: u64 },

    #[error("GPU budget exceeded during dream: {usage:.1}% > {max:.1}%")]
    GpuBudgetExceeded { usage: f32, max: f32 },

    #[error("Failed to signal wake: {reason}")]
    SignalFailed { reason: String },

    #[error("Wake controller not initialized")]
    NotInitialized,
}

/// Wake controller state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WakeState {
    /// Controller idle, no dream active
    Idle,
    /// Dream in progress, ready to wake
    Dreaming,
    /// Wake signal sent, waiting for completion
    Waking,
    /// Wake completed, processing cleanup
    Completing,
}

/// Resource usage snapshot during dream.
#[derive(Debug, Clone, Default)]
pub struct ResourceSnapshot {
    /// GPU utilization [0.0, 1.0]
    pub gpu_usage: f32,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU usage estimate [0.0, 1.0]
    pub cpu_usage: f32,
    /// Timestamp of snapshot
    pub timestamp: Instant,
}

/// Fast wake controller for dream interruption.
///
/// Guarantees <100ms wake latency through:
/// 1. Atomic interrupt flags checked at each processing step
/// 2. Pre-allocated cleanup state
/// 3. Non-blocking signal propagation
///
/// # Constitution Compliance
///
/// - Wake latency: <100ms (enforced, logged on violation)
/// - GPU budget: <30% (monitored, triggers wake on violation)
/// - Abort on query: true (external query -> immediate wake)
#[derive(Debug)]
pub struct WakeController {
    /// Current state
    state: Arc<std::sync::RwLock<WakeState>>,

    /// Interrupt flag (shared with dream phases)
    interrupt_flag: Arc<AtomicBool>,

    /// Wake reason channel
    wake_sender: watch::Sender<Option<WakeReason>>,
    wake_receiver: watch::Receiver<Option<WakeReason>>,

    /// Wake start time (for latency measurement)
    wake_start: Arc<std::sync::RwLock<Option<Instant>>>,

    /// Wake completion time
    wake_complete: Arc<std::sync::RwLock<Option<Instant>>>,

    /// Maximum allowed latency (Constitution: <100ms)
    max_latency: Duration,

    /// GPU monitor for budget enforcement
    gpu_monitor: Arc<std::sync::RwLock<GpuMonitor>>,

    /// Maximum GPU usage during dream (Constitution: 30%)
    max_gpu_usage: f32,

    /// GPU check interval
    gpu_check_interval: Duration,

    /// Last GPU check time (millis since process start)
    last_gpu_check: Arc<AtomicU64>,

    /// Wake count for statistics
    wake_count: AtomicU64,

    /// Latency violation count
    latency_violations: AtomicU64,
}

impl WakeController {
    /// Create a new wake controller with constitution defaults.
    pub fn new() -> Self {
        let (wake_sender, wake_receiver) = watch::channel(None);

        Self {
            state: Arc::new(std::sync::RwLock::new(WakeState::Idle)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            wake_sender,
            wake_receiver,
            wake_start: Arc::new(std::sync::RwLock::new(None)),
            wake_complete: Arc::new(std::sync::RwLock::new(None)),
            max_latency: constants::MAX_WAKE_LATENCY,
            gpu_monitor: Arc::new(std::sync::RwLock::new(GpuMonitor::new())),
            max_gpu_usage: constants::MAX_GPU_USAGE,
            gpu_check_interval: Duration::from_millis(100),
            last_gpu_check: Arc::new(AtomicU64::new(0)),
            wake_count: AtomicU64::new(0),
            latency_violations: AtomicU64::new(0),
        }
    }

    /// Get the shared interrupt flag for dream phases.
    pub fn interrupt_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.interrupt_flag)
    }

    /// Prepare controller for a new dream cycle.
    pub fn prepare_for_dream(&self) {
        let mut state = self.state.write().expect("Lock poisoned");
        *state = WakeState::Dreaming;
        drop(state);

        // Reset interrupt flag
        self.interrupt_flag.store(false, Ordering::SeqCst);

        // Clear wake times
        *self.wake_start.write().expect("Lock poisoned") = None;
        *self.wake_complete.write().expect("Lock poisoned") = None;

        // Send None to clear any previous wake reason
        let _ = self.wake_sender.send(None);

        debug!("Wake controller prepared for dream cycle");
    }

    /// Signal wake from dream state.
    ///
    /// Returns immediately after signaling; actual wake completes asynchronously.
    /// Measures latency from signal to completion.
    pub fn signal_wake(&self, reason: WakeReason) -> Result<(), WakeError> {
        let current_state = *self.state.read().expect("Lock poisoned");

        if current_state != WakeState::Dreaming {
            debug!("Wake signal ignored: not in dreaming state ({:?})", current_state);
            return Ok(());
        }

        // Record wake start time
        {
            let mut wake_start = self.wake_start.write().expect("Lock poisoned");
            *wake_start = Some(Instant::now());
        }

        // Update state
        {
            let mut state = self.state.write().expect("Lock poisoned");
            *state = WakeState::Waking;
        }

        // Set interrupt flag (checked by all phases)
        self.interrupt_flag.store(true, Ordering::SeqCst);

        // Send wake reason through channel
        self.wake_sender
            .send(Some(reason))
            .map_err(|_| WakeError::SignalFailed {
                reason: "Channel closed".to_string(),
            })?;

        info!("Wake signal sent: {:?}", reason);

        Ok(())
    }

    /// Mark wake as complete and measure latency.
    ///
    /// # Returns
    ///
    /// The measured wake latency. Returns error if latency > 100ms.
    ///
    /// # Errors
    ///
    /// Returns `WakeError::LatencyViolation` if latency exceeds constitution limit.
    pub fn complete_wake(&self) -> Result<Duration, WakeError> {
        let wake_time = Instant::now();

        // Record completion time
        {
            let mut wake_complete = self.wake_complete.write().expect("Lock poisoned");
            *wake_complete = Some(wake_time);
        }

        // Calculate latency
        let latency = {
            let wake_start = self.wake_start.read().expect("Lock poisoned");
            wake_start
                .map(|start| wake_time.duration_since(start))
                .unwrap_or(Duration::ZERO)
        };

        // Check latency violation
        if latency > self.max_latency {
            self.latency_violations.fetch_add(1, Ordering::Relaxed);
            error!(
                "CONSTITUTION VIOLATION: Wake latency {:?} > {:?} (max allowed)",
                latency, self.max_latency
            );
            return Err(WakeError::LatencyViolation {
                actual_ms: latency.as_millis() as u64,
                max_ms: self.max_latency.as_millis() as u64,
            });
        }

        // Update state
        {
            let mut state = self.state.write().expect("Lock poisoned");
            *state = WakeState::Completing;
        }

        self.wake_count.fetch_add(1, Ordering::Relaxed);

        info!("Wake completed in {:?}", latency);

        Ok(latency)
    }

    /// Reset controller to idle state.
    pub fn reset(&self) {
        let mut state = self.state.write().expect("Lock poisoned");
        *state = WakeState::Idle;
        drop(state);

        self.interrupt_flag.store(false, Ordering::SeqCst);
        *self.wake_start.write().expect("Lock poisoned") = None;
        *self.wake_complete.write().expect("Lock poisoned") = None;

        let _ = self.wake_sender.send(None);

        debug!("Wake controller reset to idle");
    }

    /// Check GPU usage and signal wake if over budget.
    ///
    /// Should be called periodically during dream.
    pub fn check_gpu_budget(&self) -> Result<(), WakeError> {
        // Rate limit checks using monotonic counter
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last_check = self.last_gpu_check.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last_check) < self.gpu_check_interval.as_millis() as u64 {
            return Ok(());
        }
        self.last_gpu_check.store(now_ms, Ordering::Relaxed);

        let usage = self.gpu_monitor.read().expect("Lock poisoned").get_usage();

        if usage > self.max_gpu_usage {
            warn!(
                "GPU usage exceeded budget: {:.1}% > {:.1}%",
                usage * 100.0,
                self.max_gpu_usage * 100.0
            );

            // Signal wake due to GPU overload
            self.signal_wake(WakeReason::GpuOverBudget)?;

            return Err(WakeError::GpuBudgetExceeded {
                usage: usage * 100.0,
                max: self.max_gpu_usage * 100.0,
            });
        }

        Ok(())
    }

    /// Get current resource snapshot.
    pub fn get_resource_snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot {
            gpu_usage: self.gpu_monitor.read().expect("Lock poisoned").get_usage(),
            memory_bytes: 0, // Future: Implement memory tracking
            cpu_usage: 0.0,  // Future: Implement CPU tracking
            timestamp: Instant::now(),
        }
    }

    /// Subscribe to wake events.
    pub fn subscribe(&self) -> watch::Receiver<Option<WakeReason>> {
        self.wake_receiver.clone()
    }

    /// Get current state.
    pub fn state(&self) -> WakeState {
        *self.state.read().expect("Lock poisoned")
    }

    /// Check if currently dreaming.
    pub fn is_dreaming(&self) -> bool {
        *self.state.read().expect("Lock poisoned") == WakeState::Dreaming
    }

    /// Check if wake has been signaled.
    pub fn is_wake_signaled(&self) -> bool {
        self.interrupt_flag.load(Ordering::SeqCst)
    }

    /// Get wake statistics.
    pub fn stats(&self) -> WakeStats {
        WakeStats {
            wake_count: self.wake_count.load(Ordering::Relaxed),
            latency_violations: self.latency_violations.load(Ordering::Relaxed),
            max_latency: self.max_latency,
            max_gpu_usage: self.max_gpu_usage,
        }
    }

    /// Update GPU usage for testing.
    pub fn set_gpu_usage(&self, usage: f32) {
        self.gpu_monitor.write().expect("Lock poisoned").set_simulated_usage(usage);
    }
}

impl Default for WakeController {
    fn default() -> Self {
        Self::new()
    }
}

/// Wake statistics.
#[derive(Debug, Clone)]
pub struct WakeStats {
    /// Total wake events
    pub wake_count: u64,
    /// Latency violations (>100ms)
    pub latency_violations: u64,
    /// Maximum allowed latency
    pub max_latency: Duration,
    /// Maximum GPU usage allowed
    pub max_gpu_usage: f32,
}

/// Handle for external systems to signal wake.
///
/// Lightweight clone of wake signaling capability.
#[derive(Clone)]
pub struct WakeHandle {
    interrupt_flag: Arc<AtomicBool>,
    wake_sender: watch::Sender<Option<WakeReason>>,
}

impl WakeHandle {
    /// Create from wake controller.
    pub fn from_controller(controller: &WakeController) -> Self {
        Self {
            interrupt_flag: controller.interrupt_flag(),
            wake_sender: controller.wake_sender.clone(),
        }
    }

    /// Signal immediate wake.
    pub fn wake(&self, reason: WakeReason) {
        self.interrupt_flag.store(true, Ordering::SeqCst);
        let _ = self.wake_sender.send(Some(reason));
    }

    /// Check if wake was signaled.
    pub fn is_signaled(&self) -> bool {
        self.interrupt_flag.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_wake_controller_creation() {
        let controller = WakeController::new();
        assert_eq!(controller.state(), WakeState::Idle);
        assert!(!controller.is_dreaming());
        assert!(controller.max_latency.as_millis() < 100, "Must be <100ms per Constitution");
    }

    #[test]
    fn test_prepare_for_dream() {
        let controller = WakeController::new();
        controller.prepare_for_dream();
        assert_eq!(controller.state(), WakeState::Dreaming);
        assert!(controller.is_dreaming());
        assert!(!controller.is_wake_signaled());
    }

    #[test]
    fn test_signal_wake() {
        let controller = WakeController::new();
        controller.prepare_for_dream();
        controller.signal_wake(WakeReason::ExternalQuery).unwrap();
        assert_eq!(controller.state(), WakeState::Waking);
        assert!(controller.is_wake_signaled());
    }

    #[test]
    fn test_wake_latency_success() {
        let controller = WakeController::new();
        controller.prepare_for_dream();
        controller.signal_wake(WakeReason::ExternalQuery).unwrap();

        // Complete immediately (should be well under 100ms)
        let latency = controller.complete_wake().unwrap();
        assert!(latency < Duration::from_millis(100), "Latency {:?} must be <100ms", latency);
        assert_eq!(controller.stats().wake_count, 1);
        assert_eq!(controller.stats().latency_violations, 0);
    }

    #[test]
    fn test_reset() {
        let controller = WakeController::new();
        controller.prepare_for_dream();
        controller.signal_wake(WakeReason::ManualAbort).unwrap();
        controller.reset();
        assert_eq!(controller.state(), WakeState::Idle);
        assert!(!controller.is_wake_signaled());
    }

    #[test]
    fn test_gpu_budget_check_ok() {
        let controller = WakeController::new();
        controller.prepare_for_dream();

        // Set GPU usage below budget (30%)
        controller.set_gpu_usage(0.2);

        // Reset rate limiter
        controller.last_gpu_check.store(0, Ordering::Relaxed);

        // Should pass
        controller.check_gpu_budget().unwrap();
        assert!(controller.is_dreaming());
    }

    #[test]
    fn test_gpu_budget_exceeded() {
        let controller = WakeController::new();
        controller.prepare_for_dream();

        // Set GPU usage above budget (30%)
        controller.set_gpu_usage(0.5);

        // Reset rate limiter
        controller.last_gpu_check.store(0, Ordering::Relaxed);

        // Should fail and signal wake
        let result = controller.check_gpu_budget();
        assert!(matches!(result, Err(WakeError::GpuBudgetExceeded { .. })));
        assert!(controller.is_wake_signaled());
    }

    #[test]
    fn test_wake_handle() {
        let controller = WakeController::new();
        let handle = WakeHandle::from_controller(&controller);

        controller.prepare_for_dream();
        assert!(!handle.is_signaled());

        handle.wake(WakeReason::ExternalQuery);
        assert!(handle.is_signaled());
        assert!(controller.is_wake_signaled());
    }

    #[test]
    fn test_interrupt_flag_sharing() {
        let controller = WakeController::new();
        let flag = controller.interrupt_flag();

        controller.prepare_for_dream();

        // Simulate phase checking flag
        assert!(!flag.load(Ordering::SeqCst));

        // Signal wake
        controller.signal_wake(WakeReason::ManualAbort).unwrap();

        // Phase should see interrupt
        assert!(flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_wake_subscription() {
        let controller = WakeController::new();
        let mut receiver = controller.subscribe();

        controller.prepare_for_dream();

        // Initially None
        assert!(receiver.borrow().is_none());

        // Signal wake
        controller.signal_wake(WakeReason::ExternalQuery).unwrap();

        // Wait for change
        receiver.changed().await.unwrap();

        // Should receive wake reason
        assert_eq!(*receiver.borrow(), Some(WakeReason::ExternalQuery));
    }
}
```

### 3.2 CREATE: `crates/context-graph-core/src/dream/mcp_events.rs`

```rust
//! MCP Event Integration for Dream Layer
//!
//! Defines MCP events for broadcasting dream state to the GWT workspace.
//! These events enable other subsystems to react to dream cycles.
//!
//! ## Event Categories
//!
//! 1. **Lifecycle Events**: DreamCycleStarted, DreamCycleCompleted
//! 2. **Phase Events**: NremPhaseCompleted, RemPhaseCompleted
//! 3. **Discovery Events**: BlindSpotDiscovered, ShortcutCreated
//! 4. **Resource Events**: GpuBudgetWarning, WakeTriggered

use std::time::Duration;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::ExtendedTriggerReason;
use super::WakeReason;

/// Base trait for dream events.
pub trait DreamEvent: Serialize + Clone + Send + Sync {
    /// Event type identifier for routing.
    fn event_type(&self) -> &'static str;

    /// Session ID for correlation.
    fn session_id(&self) -> Uuid;

    /// Timestamp in milliseconds since epoch.
    fn timestamp_ms(&self) -> u64;
}

/// Dream cycle started event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamCycleStarted {
    pub session_id: Uuid,
    pub trigger_reason: String,
    pub timestamp_ms: u64,
    pub expected_nrem_duration_ms: u64,
    pub expected_rem_duration_ms: u64,
}

impl DreamEvent for DreamCycleStarted {
    fn event_type(&self) -> &'static str { "dream_cycle_started" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

impl DreamCycleStarted {
    pub fn new(trigger_reason: ExtendedTriggerReason) -> Self {
        Self {
            session_id: Uuid::new_v4(),
            trigger_reason: trigger_reason.to_string(),
            timestamp_ms: current_timestamp_ms(),
            expected_nrem_duration_ms: 180_000, // 3 min per Constitution
            expected_rem_duration_ms: 120_000,  // 2 min per Constitution
        }
    }

    pub fn with_session_id(session_id: Uuid, trigger_reason: ExtendedTriggerReason) -> Self {
        Self {
            session_id,
            trigger_reason: trigger_reason.to_string(),
            timestamp_ms: current_timestamp_ms(),
            expected_nrem_duration_ms: 180_000,
            expected_rem_duration_ms: 120_000,
        }
    }
}

/// NREM phase completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NremPhaseCompleted {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub memories_replayed: usize,
    pub edges_strengthened: usize,
    pub edges_pruned: usize,
    pub duration_ms: u64,
    pub compression_ratio: f32,
}

impl DreamEvent for NremPhaseCompleted {
    fn event_type(&self) -> &'static str { "nrem_phase_completed" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

impl NremPhaseCompleted {
    pub fn new(
        session_id: Uuid,
        memories_replayed: usize,
        edges_strengthened: usize,
        edges_pruned: usize,
        duration: Duration,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            memories_replayed,
            edges_strengthened,
            edges_pruned,
            duration_ms: duration.as_millis() as u64,
            compression_ratio: if memories_replayed > 0 {
                edges_pruned as f32 / memories_replayed as f32
            } else {
                0.0
            },
        }
    }
}

/// REM phase completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemPhaseCompleted {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub queries_generated: usize,
    pub blind_spots_found: usize,
    pub walk_distance: f32,
    pub duration_ms: u64,
    pub average_semantic_leap: f32,
}

impl DreamEvent for RemPhaseCompleted {
    fn event_type(&self) -> &'static str { "rem_phase_completed" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

impl RemPhaseCompleted {
    pub fn new(
        session_id: Uuid,
        queries_generated: usize,
        blind_spots_found: usize,
        walk_distance: f32,
        duration: Duration,
        average_semantic_leap: f32,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            queries_generated,
            blind_spots_found,
            walk_distance,
            duration_ms: duration.as_millis() as u64,
            average_semantic_leap,
        }
    }
}

/// Dream cycle completed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamCycleCompleted {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub completed: bool,
    pub wake_reason: String,
    pub shortcuts_created: usize,
    pub total_duration_ms: u64,
    pub wake_latency_ms: u64,
}

impl DreamEvent for DreamCycleCompleted {
    fn event_type(&self) -> &'static str { "dream_cycle_completed" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

impl DreamCycleCompleted {
    pub fn new(
        session_id: Uuid,
        completed: bool,
        wake_reason: WakeReason,
        shortcuts_created: usize,
        total_duration: Duration,
        wake_latency: Duration,
    ) -> Self {
        Self {
            session_id,
            timestamp_ms: current_timestamp_ms(),
            completed,
            wake_reason: wake_reason.to_string(),
            shortcuts_created,
            total_duration_ms: total_duration.as_millis() as u64,
            wake_latency_ms: wake_latency.as_millis() as u64,
        }
    }
}

/// Blind spot discovered event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindSpotDiscovered {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub blind_spot_id: Uuid,
    pub poincare_position: Vec<f32>, // Variable length to avoid serde issues with [f32; 64]
    pub semantic_distance: f32,
    pub confidence: f32,
    pub discovery_step: usize,
}

impl DreamEvent for BlindSpotDiscovered {
    fn event_type(&self) -> &'static str { "blind_spot_discovered" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

/// Shortcut edge created event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortcutCreated {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub hop_count: usize,
    pub combined_weight: f32,
    pub traversal_count: usize,
}

impl DreamEvent for ShortcutCreated {
    fn event_type(&self) -> &'static str { "shortcut_created" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

/// GPU budget warning event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBudgetWarning {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub current_usage: f32,
    pub budget: f32,
    pub action_taken: String,
}

impl DreamEvent for GpuBudgetWarning {
    fn event_type(&self) -> &'static str { "gpu_budget_warning" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

/// Wake triggered event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeTriggered {
    pub session_id: Uuid,
    pub timestamp_ms: u64,
    pub reason: String,
    pub phase: String,
    pub latency_ms: u64,
}

impl DreamEvent for WakeTriggered {
    fn event_type(&self) -> &'static str { "wake_triggered" }
    fn session_id(&self) -> Uuid { self.session_id }
    fn timestamp_ms(&self) -> u64 { self.timestamp_ms }
}

/// MCP event broadcaster interface.
///
/// Implementations should connect to actual MCP transport.
pub trait DreamEventBroadcaster: Send + Sync {
    /// Broadcast a dream event.
    fn broadcast<E: DreamEvent>(&self, event: &E) -> Result<(), BroadcastError>;

    /// Check if broadcaster is connected.
    fn is_connected(&self) -> bool;
}

/// Broadcast error type.
#[derive(Debug, thiserror::Error)]
pub enum BroadcastError {
    #[error("Not connected to MCP")]
    NotConnected,

    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Transport error: {0}")]
    TransportError(String),
}

/// No-op broadcaster for testing.
#[derive(Debug, Default)]
pub struct NoOpBroadcaster;

impl DreamEventBroadcaster for NoOpBroadcaster {
    fn broadcast<E: DreamEvent>(&self, _event: &E) -> Result<(), BroadcastError> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        false
    }
}

/// Logging broadcaster for development.
#[derive(Debug, Default)]
pub struct LoggingBroadcaster;

impl DreamEventBroadcaster for LoggingBroadcaster {
    fn broadcast<E: DreamEvent>(&self, event: &E) -> Result<(), BroadcastError> {
        let json = serde_json::to_string(event)
            .map_err(|e| BroadcastError::SerializationFailed(e.to_string()))?;
        tracing::info!(target: "mcp_events", event_type = %event.event_type(), "{}", json);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        true
    }
}

/// Get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_cycle_started() {
        let event = DreamCycleStarted::new(ExtendedTriggerReason::HighEntropy);

        assert_eq!(event.event_type(), "dream_cycle_started");
        assert_eq!(event.trigger_reason, "high_entropy");
        assert!(event.timestamp_ms > 0);
        assert!(!event.session_id.is_nil());
    }

    #[test]
    fn test_dream_cycle_completed() {
        let session_id = Uuid::new_v4();
        let event = DreamCycleCompleted::new(
            session_id,
            true,
            WakeReason::CycleComplete,
            5,
            Duration::from_secs(300),
            Duration::from_millis(50),
        );

        assert_eq!(event.event_type(), "dream_cycle_completed");
        assert!(event.completed);
        assert_eq!(event.shortcuts_created, 5);
        assert_eq!(event.total_duration_ms, 300_000);
        assert_eq!(event.wake_latency_ms, 50);
    }

    #[test]
    fn test_nrem_phase_completed() {
        let session_id = Uuid::new_v4();
        let event = NremPhaseCompleted::new(
            session_id,
            100,  // memories replayed
            75,   // edges strengthened
            10,   // edges pruned
            Duration::from_secs(180),
        );

        assert_eq!(event.event_type(), "nrem_phase_completed");
        assert_eq!(event.memories_replayed, 100);
        assert_eq!(event.edges_strengthened, 75);
        assert_eq!(event.edges_pruned, 10);
        assert_eq!(event.duration_ms, 180_000);
    }

    #[test]
    fn test_rem_phase_completed() {
        let session_id = Uuid::new_v4();
        let event = RemPhaseCompleted::new(
            session_id,
            50,   // queries generated
            3,    // blind spots found
            0.8,  // walk distance
            Duration::from_secs(120),
            0.75, // average semantic leap
        );

        assert_eq!(event.event_type(), "rem_phase_completed");
        assert_eq!(event.queries_generated, 50);
        assert_eq!(event.blind_spots_found, 3);
        assert_eq!(event.duration_ms, 120_000);
    }

    #[test]
    fn test_logging_broadcaster() {
        let broadcaster = LoggingBroadcaster;
        let event = DreamCycleStarted::new(ExtendedTriggerReason::Manual);

        assert!(broadcaster.is_connected());
        broadcaster.broadcast(&event).unwrap();
    }

    #[test]
    fn test_noop_broadcaster() {
        let broadcaster = NoOpBroadcaster;
        let event = DreamCycleStarted::new(ExtendedTriggerReason::IdleTimeout);

        assert!(!broadcaster.is_connected());
        broadcaster.broadcast(&event).unwrap();
    }

    #[test]
    fn test_events_serialize_to_json() {
        let session_id = Uuid::new_v4();

        // All events must serialize without error
        let events: Vec<Box<dyn erased_serde::Serialize>> = vec![
            Box::new(DreamCycleStarted::new(ExtendedTriggerReason::Manual)),
            Box::new(DreamCycleCompleted::new(
                session_id, true, WakeReason::CycleComplete, 0, Duration::ZERO, Duration::ZERO
            )),
            Box::new(NremPhaseCompleted::new(session_id, 0, 0, 0, Duration::ZERO)),
            Box::new(RemPhaseCompleted::new(session_id, 0, 0, 0.0, Duration::ZERO, 0.0)),
            Box::new(WakeTriggered {
                session_id,
                timestamp_ms: current_timestamp_ms(),
                reason: "test".to_string(),
                phase: "nrem".to_string(),
                latency_ms: 50,
            }),
        ];

        for event in events {
            let json = serde_json::to_string(&event);
            assert!(json.is_ok(), "Event serialization failed: {:?}", json.err());
        }
    }
}
```

### 3.3 MODIFY: `crates/context-graph-core/src/dream/mod.rs`

Add these lines after line 56 (after `pub mod types;`):

```rust
pub mod mcp_events;
pub mod wake_controller;
```

Add these re-exports after line 103 (after the existing re-exports):

```rust
pub use mcp_events::{
    BlindSpotDiscovered,
    BroadcastError,
    DreamCycleCompleted,
    DreamCycleStarted,
    DreamEvent,
    DreamEventBroadcaster,
    GpuBudgetWarning,
    LoggingBroadcaster,
    NoOpBroadcaster,
    NremPhaseCompleted,
    RemPhaseCompleted,
    ShortcutCreated,
    WakeTriggered,
};
pub use wake_controller::{
    ResourceSnapshot,
    WakeController,
    WakeError,
    WakeHandle,
    WakeState,
    WakeStats,
};
```

### 3.4 DO NOT Modify Cargo.toml

The current dependencies are sufficient:
- `tokio::sync::watch` is already available from `tokio.workspace`
- `std::sync::RwLock` is used instead of `parking_lot` (standard library)
- `erased-serde` is NOT needed - we use generic `<E: DreamEvent>` instead

---

## 4. Definition of Done

### 4.1 Type Signatures (Exact)

```rust
// wake_controller.rs
pub struct WakeController { /* internal */ }
impl WakeController {
    pub fn new() -> Self;
    pub fn interrupt_flag(&self) -> Arc<AtomicBool>;
    pub fn prepare_for_dream(&self);
    pub fn signal_wake(&self, reason: WakeReason) -> Result<(), WakeError>;
    pub fn complete_wake(&self) -> Result<Duration, WakeError>;
    pub fn reset(&self);
    pub fn check_gpu_budget(&self) -> Result<(), WakeError>;
    pub fn get_resource_snapshot(&self) -> ResourceSnapshot;
    pub fn subscribe(&self) -> watch::Receiver<Option<WakeReason>>;
    pub fn state(&self) -> WakeState;
    pub fn is_dreaming(&self) -> bool;
    pub fn is_wake_signaled(&self) -> bool;
    pub fn stats(&self) -> WakeStats;
    pub fn set_gpu_usage(&self, usage: f32);
}

pub struct WakeHandle { /* internal */ }
impl WakeHandle {
    pub fn from_controller(controller: &WakeController) -> Self;
    pub fn wake(&self, reason: WakeReason);
    pub fn is_signaled(&self) -> bool;
}

pub trait DreamEventBroadcaster: Send + Sync {
    fn broadcast<E: DreamEvent>(&self, event: &E) -> Result<(), BroadcastError>;
    fn is_connected(&self) -> bool;
}

pub enum WakeState { Idle, Dreaming, Waking, Completing }
pub enum WakeError { LatencyViolation { .. }, GpuBudgetExceeded { .. }, SignalFailed { .. }, NotInitialized }
```

### 4.2 Validation Commands

```bash
# STEP 1: Build
cargo build -p context-graph-core 2>&1 | tee /tmp/build.log
# Expected: Compiles with NO errors

# STEP 2: Run wake_controller tests
cargo test -p context-graph-core wake_controller -- --nocapture 2>&1 | tee /tmp/wake_tests.log
# Expected: All tests pass

# STEP 3: Run mcp_events tests
cargo test -p context-graph-core mcp_events -- --nocapture 2>&1 | tee /tmp/mcp_tests.log
# Expected: All tests pass

# STEP 4: Clippy check
cargo clippy -p context-graph-core -- -D warnings 2>&1 | tee /tmp/clippy.log
# Expected: No errors (warnings may exist from other modules)

# STEP 5: Verify exports
cargo doc -p context-graph-core --no-deps 2>&1 | tee /tmp/doc.log
# Expected: Documentation builds successfully
```

---

## 5. Full State Verification Protocol

### 5.1 Source of Truth

| Component | Source of Truth | Verification Method |
|-----------|-----------------|---------------------|
| WakeController state | `WakeController.state()` | Call method, verify `WakeState` enum |
| Interrupt flag | `AtomicBool` via `interrupt_flag()` | `flag.load(Ordering::SeqCst)` |
| Wake latency | Return value from `complete_wake()` | Check Duration < 100ms |
| GPU budget | `check_gpu_budget()` result | Should error if usage > 30% |
| MCP events | JSON serialization output | Verify all fields present |

### 5.2 Execute & Inspect Protocol

After each operation:
1. Call the method
2. Immediately read the source of truth
3. Compare expected vs actual
4. Log both states

Example verification:
```rust
// BEFORE
println!("BEFORE: state={:?}, signaled={}", controller.state(), controller.is_wake_signaled());

// ACTION
controller.signal_wake(WakeReason::ExternalQuery)?;

// AFTER
println!("AFTER: state={:?}, signaled={}", controller.state(), controller.is_wake_signaled());

// VERIFY
assert_eq!(controller.state(), WakeState::Waking);
assert!(controller.is_wake_signaled());
```

### 5.3 Edge Case Audit (3 Required)

#### Edge Case 1: Wake During Idle State
```rust
// Setup: Controller in Idle state (not dreaming)
let controller = WakeController::new();
println!("STATE BEFORE: {:?}", controller.state()); // Idle

// Action: Try to signal wake when not dreaming
let result = controller.signal_wake(WakeReason::ExternalQuery);
println!("STATE AFTER: {:?}", controller.state()); // Still Idle

// Expected: No error, state unchanged (wake ignored)
assert!(result.is_ok());
assert_eq!(controller.state(), WakeState::Idle);
```

#### Edge Case 2: GPU Budget Exactly at 30%
```rust
// Setup
let controller = WakeController::new();
controller.prepare_for_dream();
controller.set_gpu_usage(0.30); // Exactly at threshold
controller.last_gpu_check.store(0, Ordering::Relaxed);

println!("GPU: 30%, STATE BEFORE: {:?}", controller.state());

// Action: Check GPU budget
let result = controller.check_gpu_budget();

println!("STATE AFTER: {:?}, signaled: {}", controller.state(), controller.is_wake_signaled());

// Expected: Should trigger wake (>= threshold)
assert!(matches!(result, Err(WakeError::GpuBudgetExceeded { .. })));
```

#### Edge Case 3: Double Wake Signal
```rust
// Setup
let controller = WakeController::new();
controller.prepare_for_dream();

// First wake
controller.signal_wake(WakeReason::ExternalQuery)?;
println!("STATE AFTER FIRST WAKE: {:?}", controller.state()); // Waking

// Second wake attempt (should be ignored)
let result = controller.signal_wake(WakeReason::ManualAbort);
println!("STATE AFTER SECOND WAKE: {:?}", controller.state()); // Still Waking

// Expected: No error, second wake ignored
assert!(result.is_ok());
assert_eq!(controller.state(), WakeState::Waking);
```

### 5.4 Evidence of Success Log Format

```
=== WAKE CONTROLLER VERIFICATION ===
Timestamp: 2026-01-12T10:00:00Z

TEST: test_wake_controller_creation
  BEFORE: state=Idle, is_dreaming=false
  ACTION: WakeController::new()
  AFTER: state=Idle, is_dreaming=false, max_latency=99ms
  RESULT: PASS

TEST: test_signal_wake
  BEFORE: state=Dreaming, signaled=false
  ACTION: signal_wake(ExternalQuery)
  AFTER: state=Waking, signaled=true
  RESULT: PASS

TEST: test_wake_latency_success
  BEFORE: state=Waking
  ACTION: complete_wake()
  AFTER: state=Completing, latency=1ms, violations=0
  RESULT: PASS (latency < 100ms)

TEST: test_gpu_budget_exceeded
  BEFORE: state=Dreaming, gpu_usage=50%
  ACTION: check_gpu_budget()
  AFTER: state=Waking, signaled=true
  ERROR: GpuBudgetExceeded { usage: 50.0, max: 30.0 }
  RESULT: PASS (correctly detected budget violation)

=== ALL TESTS PASSED ===
```

---

## 6. Manual Testing Checklist

Run these commands and verify outputs:

### 6.1 Build Verification
```bash
cd /home/cabdru/contextgraph
cargo build -p context-graph-core 2>&1
# EXPECTED: "Finished" with no errors
```

### 6.2 Test Execution
```bash
cargo test -p context-graph-core wake_controller -- --nocapture
# EXPECTED: "test result: ok" with all tests passing
```

### 6.3 Integration Test (Synthetic Data)

Create and run a test that exercises the full wake flow:

```rust
#[tokio::test]
async fn integration_test_full_wake_cycle() {
    // SYNTHETIC INPUT
    let controller = WakeController::new();

    // 1. Prepare for dream
    controller.prepare_for_dream();
    assert_eq!(controller.state(), WakeState::Dreaming);
    println!("Step 1 PASS: state=Dreaming");

    // 2. Simulate some dream processing
    tokio::time::sleep(Duration::from_millis(10)).await;

    // 3. Signal external query wake
    controller.signal_wake(WakeReason::ExternalQuery).unwrap();
    assert!(controller.is_wake_signaled());
    println!("Step 2 PASS: wake signaled");

    // 4. Complete wake and measure latency
    let latency = controller.complete_wake().unwrap();
    assert!(latency < Duration::from_millis(100));
    println!("Step 3 PASS: latency={:?} < 100ms", latency);

    // 5. Reset and verify
    controller.reset();
    assert_eq!(controller.state(), WakeState::Idle);
    println!("Step 4 PASS: reset to Idle");

    // FINAL VERIFICATION
    let stats = controller.stats();
    assert_eq!(stats.wake_count, 1);
    assert_eq!(stats.latency_violations, 0);
    println!("INTEGRATION TEST PASSED: wake_count={}, violations={}", stats.wake_count, stats.latency_violations);
}
```

---

## 7. Traceability

| Requirement | Constitution Ref | Test |
|-------------|-----------------|------|
| Wake latency < 100ms | `dream.constraints.wake` (line 273) | `test_wake_latency_success` |
| GPU budget < 30% | `dream.constraints.gpu` (line 273) | `test_gpu_budget_exceeded`, `test_gpu_budget_check_ok` |
| Abort on query | `dream.constraints.abort_on_query` (line 273) | `test_signal_wake`, `test_interrupt_flag_sharing` |
| Interrupt flag sharing | `dream.wake` (line 273) | `test_interrupt_flag_sharing`, `test_wake_handle` |

---

## 8. Absolute Rules

1. **NO BACKWARDS COMPATIBILITY WORKAROUNDS** - Code must work or fail fast with clear errors
2. **NO MOCK DATA IN TESTS** - Use real UUIDs, real time, real state
3. **FAIL FAST** - All errors must be logged with context and propagated
4. **VERIFY SOURCE OF TRUTH** - After every operation, read actual state to confirm
5. **NO PARKING_LOT DEPENDENCY** - Use `std::sync::RwLock` instead
6. **NO ERASED-SERDE DEPENDENCY** - Use generic `<E: DreamEvent>` pattern

---

## 9. Common Pitfalls to Avoid

1. **Do NOT use `parking_lot::RwLock`** - It's not in workspace, use `std::sync::RwLock` with `.expect("Lock poisoned")` for fail-fast behavior
2. **Do NOT add new dependencies to Cargo.toml** - All needed deps already exist
3. **Do NOT modify `WakeReason` enum** - It already exists in `mod.rs`
4. **Do NOT modify `GpuMonitor`** - It already exists in `triggers.rs`
5. **Do NOT create workarounds for failing tests** - Fix the root cause instead
