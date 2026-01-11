# TASK-DREAM-P0-006: Wake Controller and MCP Integration

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-DREAM-P0-006 |
| **Spec Ref** | SPEC-DREAM-001 |
| **Layer** | 3 (Surface/Integration) |
| **Priority** | P0 - Critical |
| **Effort** | 4 hours |
| **Dependencies** | TASK-DREAM-P0-003, TASK-DREAM-P0-004, TASK-DREAM-P0-005 |
| **Blocks** | None (Final task in chain) |

---

## 1. Objective

Implement the Wake Controller for managing dream cycle interruption with guaranteed <100ms latency, GPU budget enforcement during dreams, and MCP event integration for broadcasting dream state to the workspace.

This task completes the Dream Layer by:
1. Implementing fast wake transition (<100ms) per Constitution requirement
2. Enforcing GPU budget (<30%) during dream cycles
3. Wiring all dream events to MCP for GWT workspace integration
4. Integrating all components into DreamController

---

## 2. Input Context Files

```yaml
must_read:
  - path: crates/context-graph-core/src/dream/controller.rs
    purpose: Existing DreamController for integration
  - path: crates/context-graph-core/src/dream/nrem.rs
    purpose: NREM phase to wire with Hebbian engine
  - path: crates/context-graph-core/src/dream/rem.rs
    purpose: REM phase to wire with hyperbolic explorer
  - path: crates/context-graph-core/src/dream/scheduler.rs
    purpose: Scheduler to wire with trigger manager
  - path: crates/context-graph-core/src/dream/mod.rs
    purpose: Dream module constants and types

should_read:
  - path: crates/context-graph-mcp/src/workspace/broadcaster.rs
    purpose: MCP broadcasting interface (if exists)
  - path: crates/context-graph-core/src/gwt/mod.rs
    purpose: GWT integration points
```

---

## 3. Files to Create/Modify

### 3.1 Create: `crates/context-graph-core/src/dream/wake_controller.rs`

```rust
//! Wake Controller - Fast Dream Interruption System
//!
//! Manages wake transitions with guaranteed <100ms latency as required by
//! Constitution Section dream.wake. Provides resource monitoring and
//! clean abort handling.
//!
//! ## Constitution Reference
//!
//! - wake: <100ms latency
//! - gpu: <30% usage during dream
//! - abort_on_query: true

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use thiserror::Error;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use super::constants;
use super::triggers::GpuMonitor;
use super::WakeReason;

/// Error types for wake controller operations.
#[derive(Debug, Error)]
pub enum WakeError {
    #[error("Wake latency exceeded: {actual_ms}ms > {max_ms}ms")]
    LatencyViolation { actual_ms: u64, max_ms: u64 },

    #[error("GPU budget exceeded during dream: {usage:.1}% > {max:.1}%")]
    GpuBudgetExceeded { usage: f32, max: f32 },

    #[error("Failed to signal wake: {reason}")]
    SignalFailed { reason: String },

    #[error("Wake controller not initialized")]
    NotInitialized,
}

/// Wake controller state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug)]
pub struct WakeController {
    /// Current state
    state: Arc<RwLock<WakeState>>,

    /// Interrupt flag (shared with dream phases)
    interrupt_flag: Arc<AtomicBool>,

    /// Wake reason channel
    wake_sender: watch::Sender<Option<WakeReason>>,
    wake_receiver: watch::Receiver<Option<WakeReason>>,

    /// Wake start time (for latency measurement)
    wake_start: Arc<RwLock<Option<Instant>>>,

    /// Wake completion time
    wake_complete: Arc<RwLock<Option<Instant>>>,

    /// Maximum allowed latency (Constitution: <100ms)
    max_latency: Duration,

    /// GPU monitor for budget enforcement
    gpu_monitor: Arc<RwLock<GpuMonitor>>,

    /// Maximum GPU usage during dream (Constitution: 30%)
    max_gpu_usage: f32,

    /// GPU check interval
    gpu_check_interval: Duration,

    /// Last GPU check time
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
            state: Arc::new(RwLock::new(WakeState::Idle)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            wake_sender,
            wake_receiver,
            wake_start: Arc::new(RwLock::new(None)),
            wake_complete: Arc::new(RwLock::new(None)),
            max_latency: constants::MAX_WAKE_LATENCY,
            gpu_monitor: Arc::new(RwLock::new(GpuMonitor::new())),
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
        let mut state = self.state.write();
        *state = WakeState::Dreaming;

        // Reset interrupt flag
        self.interrupt_flag.store(false, Ordering::SeqCst);

        // Clear wake times
        *self.wake_start.write() = None;
        *self.wake_complete.write() = None;

        // Send None to clear any previous wake reason
        let _ = self.wake_sender.send(None);

        debug!("Wake controller prepared for dream cycle");
    }

    /// Signal wake from dream state.
    ///
    /// Returns immediately after signaling; actual wake completes asynchronously.
    /// Measures latency from signal to completion.
    pub fn signal_wake(&self, reason: WakeReason) -> Result<(), WakeError> {
        let current_state = *self.state.read();

        if current_state != WakeState::Dreaming {
            debug!("Wake signal ignored: not in dreaming state ({:?})", current_state);
            return Ok(());
        }

        // Record wake start time
        {
            let mut wake_start = self.wake_start.write();
            *wake_start = Some(Instant::now());
        }

        // Update state
        {
            let mut state = self.state.write();
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
    /// The measured wake latency
    pub fn complete_wake(&self) -> Result<Duration, WakeError> {
        let wake_time = Instant::now();

        // Record completion time
        {
            let mut wake_complete = self.wake_complete.write();
            *wake_complete = Some(wake_time);
        }

        // Calculate latency
        let latency = {
            let wake_start = self.wake_start.read();
            wake_start
                .map(|start| wake_time.duration_since(start))
                .unwrap_or(Duration::ZERO)
        };

        // Check latency violation
        if latency > self.max_latency {
            self.latency_violations.fetch_add(1, Ordering::Relaxed);
            error!(
                "Wake latency violation: {:?} > {:?}",
                latency, self.max_latency
            );
            return Err(WakeError::LatencyViolation {
                actual_ms: latency.as_millis() as u64,
                max_ms: self.max_latency.as_millis() as u64,
            });
        }

        // Update state
        {
            let mut state = self.state.write();
            *state = WakeState::Completing;
        }

        self.wake_count.fetch_add(1, Ordering::Relaxed);

        info!("Wake completed in {:?}", latency);

        Ok(latency)
    }

    /// Reset controller to idle state.
    pub fn reset(&self) {
        let mut state = self.state.write();
        *state = WakeState::Idle;

        self.interrupt_flag.store(false, Ordering::SeqCst);
        *self.wake_start.write() = None;
        *self.wake_complete.write() = None;

        let _ = self.wake_sender.send(None);

        debug!("Wake controller reset to idle");
    }

    /// Check GPU usage and signal wake if over budget.
    ///
    /// Should be called periodically during dream.
    pub fn check_gpu_budget(&self) -> Result<(), WakeError> {
        // Rate limit checks
        let now_ms = Instant::now().elapsed().as_millis() as u64;
        let last_check = self.last_gpu_check.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last_check) < self.gpu_check_interval.as_millis() as u64 {
            return Ok(());
        }
        self.last_gpu_check.store(now_ms, Ordering::Relaxed);

        let usage = self.gpu_monitor.read().get_usage();

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
            gpu_usage: self.gpu_monitor.read().get_usage(),
            memory_bytes: 0, // TODO: Implement memory tracking
            cpu_usage: 0.0,  // TODO: Implement CPU tracking
            timestamp: Instant::now(),
        }
    }

    /// Subscribe to wake events.
    pub fn subscribe(&self) -> watch::Receiver<Option<WakeReason>> {
        self.wake_receiver.clone()
    }

    /// Get current state.
    pub fn state(&self) -> WakeState {
        *self.state.read()
    }

    /// Check if currently dreaming.
    pub fn is_dreaming(&self) -> bool {
        *self.state.read() == WakeState::Dreaming
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
        self.gpu_monitor.write().set_simulated_usage(usage);
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

        assert!(latency < Duration::from_millis(100));
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

        // Set GPU usage below budget
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

### 3.2 Create: `crates/context-graph-core/src/dream/mcp_events.rs`

```rust
//! MCP Event Integration for Dream Layer
//!
//! Defines MCP events for broadcasting dream state to the GWT workspace.
//! These events enable other subsystems to react to dream cycles.
//!
//! ## Event Categories
//!
//! 1. **Lifecycle Events**: dream_started, dream_completed
//! 2. **Phase Events**: nrem_completed, rem_completed
//! 3. **Discovery Events**: blind_spot_discovered, shortcut_created
//! 4. **Resource Events**: gpu_budget_warning, wake_triggered

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
    fn event_type(&self) -> &'static str {
        "dream_cycle_started"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

impl DreamCycleStarted {
    pub fn new(trigger_reason: ExtendedTriggerReason) -> Self {
        Self {
            session_id: Uuid::new_v4(),
            trigger_reason: trigger_reason.to_string(),
            timestamp_ms: current_timestamp_ms(),
            expected_nrem_duration_ms: 180_000, // 3 min
            expected_rem_duration_ms: 120_000,  // 2 min
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
    fn event_type(&self) -> &'static str {
        "nrem_phase_completed"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
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
    fn event_type(&self) -> &'static str {
        "rem_phase_completed"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
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
    fn event_type(&self) -> &'static str {
        "dream_cycle_completed"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
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
    pub poincare_position: [f32; 64],
    pub semantic_distance: f32,
    pub confidence: f32,
    pub discovery_step: usize,
}

impl DreamEvent for BlindSpotDiscovered {
    fn event_type(&self) -> &'static str {
        "blind_spot_discovered"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
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
    fn event_type(&self) -> &'static str {
        "shortcut_created"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
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
    fn event_type(&self) -> &'static str {
        "gpu_budget_warning"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
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
    fn event_type(&self) -> &'static str {
        "wake_triggered"
    }

    fn session_id(&self) -> Uuid {
        self.session_id
    }

    fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

/// MCP event broadcaster interface.
///
/// Implementations should connect to actual MCP transport.
pub trait DreamEventBroadcaster: Send + Sync {
    /// Broadcast a dream event.
    fn broadcast(&self, event: &dyn erased_serde::Serialize) -> Result<(), BroadcastError>;

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
    fn broadcast(&self, _event: &dyn erased_serde::Serialize) -> Result<(), BroadcastError> {
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
    fn broadcast(&self, event: &dyn erased_serde::Serialize) -> Result<(), BroadcastError> {
        let json = serde_json::to_string(event)
            .map_err(|e| BroadcastError::SerializationFailed(e.to_string()))?;
        tracing::info!(target: "mcp_events", "{}", json);
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
}
```

### 3.3 Modify: `crates/context-graph-core/src/dream/mod.rs`

Add new module exports:

```rust
// Add after existing modules (after line 51):
pub mod mcp_events;
pub mod wake_controller;

// Add to re-exports (after line 58):
pub use mcp_events::{
    DreamCycleStarted,
    DreamCycleCompleted,
    NremPhaseCompleted,
    RemPhaseCompleted,
    BlindSpotDiscovered,
    ShortcutCreated,
    GpuBudgetWarning,
    WakeTriggered,
    DreamEvent,
    DreamEventBroadcaster,
    LoggingBroadcaster,
    NoOpBroadcaster,
};
pub use wake_controller::{
    WakeController,
    WakeHandle,
    WakeState,
    WakeStats,
    WakeError,
    ResourceSnapshot,
};
```

### 3.4 Modify: `crates/context-graph-core/src/dream/controller.rs`

Update DreamController to integrate all components:

```rust
// Add imports at top:
use super::hebbian::HebbianEngine;
use super::hyperbolic_walk::HyperbolicExplorer;
use super::triggers::GwtIntegration;
use super::wake_controller::{WakeController, WakeHandle};
use super::mcp_events::{
    DreamCycleStarted, DreamCycleCompleted, NremPhaseCompleted, RemPhaseCompleted,
    DreamEventBroadcaster, LoggingBroadcaster,
};
use super::types::{HebbianConfig, HyperbolicWalkConfig, ExtendedTriggerReason};

// Add to DreamController struct:
    /// Wake controller for interrupt handling
    wake_controller: WakeController,

    /// GWT integration for trigger management
    gwt_integration: GwtIntegration,

    /// MCP event broadcaster
    broadcaster: Box<dyn DreamEventBroadcaster>,

    /// Hebbian learning engine
    hebbian_engine: HebbianEngine,

    /// Hyperbolic explorer
    hyperbolic_explorer: HyperbolicExplorer,

// Update new() method to initialize all components:
pub fn new() -> Self {
    Self {
        // ... existing fields ...
        wake_controller: WakeController::new(),
        gwt_integration: GwtIntegration::new(),
        broadcaster: Box::new(LoggingBroadcaster),
        hebbian_engine: HebbianEngine::with_defaults(),
        hyperbolic_explorer: HyperbolicExplorer::with_defaults(),
    }
}

// Add method to get wake handle for external systems:
pub fn wake_handle(&self) -> WakeHandle {
    WakeHandle::from_controller(&self.wake_controller)
}

// Add method for external query notification (triggers wake):
pub fn on_external_query(&mut self) {
    self.gwt_integration.on_query();

    if self.is_dreaming() {
        self.wake_controller.signal_wake(WakeReason::ExternalQuery).ok();
    }
}
```

### 3.5 Update: `crates/context-graph-core/Cargo.toml`

Add required dependencies:

```toml
[dependencies]
# Existing deps...
parking_lot = "0.12"
erased-serde = "0.4"
```

---

## 4. Definition of Done

### 4.1 Type Signatures (Exact)

```rust
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
}

pub struct WakeHandle { /* internal */ }
impl WakeHandle {
    pub fn from_controller(controller: &WakeController) -> Self;
    pub fn wake(&self, reason: WakeReason);
    pub fn is_signaled(&self) -> bool;
}

pub trait DreamEventBroadcaster: Send + Sync {
    fn broadcast(&self, event: &dyn erased_serde::Serialize) -> Result<(), BroadcastError>;
    fn is_connected(&self) -> bool;
}

pub enum WakeState { Idle, Dreaming, Waking, Completing }
pub enum WakeError { LatencyViolation, GpuBudgetExceeded, SignalFailed, NotInitialized }
```

### 4.2 Validation Criteria

| Criterion | Check |
|-----------|-------|
| Compiles | `cargo build -p context-graph-core` |
| Tests pass | `cargo test -p context-graph-core dream::wake_controller` |
| Tests pass | `cargo test -p context-graph-core dream::mcp_events` |
| No clippy warnings | `cargo clippy -p context-graph-core` |
| Wake latency < 100ms | Signal to complete measured |
| GPU budget enforced | Triggers wake at >30% |
| Interrupt flag shared | All phases check same flag |
| MCP events serializable | All events serialize to JSON |

### 4.3 Test Coverage Requirements

- [ ] WakeController creation
- [ ] Prepare for dream state transition
- [ ] Signal wake sets interrupt flag
- [ ] Wake latency measurement
- [ ] Wake latency violation detection
- [ ] Reset clears all state
- [ ] GPU budget check passes when under
- [ ] GPU budget check triggers wake when over
- [ ] WakeHandle signals controller
- [ ] Interrupt flag shared between controller and phases
- [ ] Wake subscription receives events
- [ ] All MCP events serialize correctly
- [ ] LoggingBroadcaster works
- [ ] NoOpBroadcaster works

---

## 5. Implementation Notes

### 5.1 Wake Latency Guarantee

The <100ms wake latency is achieved through:

1. **Atomic interrupt flag**: All phases check `interrupt_flag.load(Ordering::SeqCst)` at every loop iteration
2. **No blocking operations**: Wake signal uses non-blocking channels
3. **Pre-allocated state**: No allocations during wake path
4. **Short-circuit evaluation**: Phases exit immediately on interrupt

### 5.2 GPU Budget Enforcement

- Check interval: 100ms (configurable)
- Budget: 30% (Constitution)
- Action: Signal wake with `WakeReason::GpuOverBudget`
- Simulated for testing; real implementation requires NVML

### 5.3 MCP Event Schema

All events follow schema:
```json
{
  "event_type": "string",
  "session_id": "uuid",
  "timestamp_ms": "u64",
  ...additional fields...
}
```

### 5.4 Constitution Compliance

- Wake latency: <100ms (enforced, measured)
- GPU budget: <30% (enforced, triggers wake)
- Abort on query: true (external query triggers immediate wake)

---

## 6. Integration Checklist

After completing this task, verify:

- [ ] DreamController uses WakeController for all interrupts
- [ ] DreamController broadcasts all lifecycle events
- [ ] NremPhase uses HebbianEngine (from TASK-003)
- [ ] RemPhase uses HyperbolicExplorer (from TASK-004)
- [ ] DreamScheduler uses TriggerManager (from TASK-005)
- [ ] External query hook wired to wake signal
- [ ] All phases check shared interrupt_flag
- [ ] GPU monitoring integrated with wake controller

---

## 7. Estimated Effort Breakdown

| Phase | Duration |
|-------|----------|
| WakeController core | 60 min |
| Latency measurement | 30 min |
| GPU budget enforcement | 20 min |
| MCP event definitions | 45 min |
| Broadcaster implementations | 20 min |
| DreamController integration | 30 min |
| Unit tests | 45 min |
| **Total** | **4 hours** |

---

## 8. Traceability

| Requirement | Test |
|-------------|------|
| REQ-DREAM-014 | test_wake_latency_success, test_wake_latency_violation |
| REQ-DREAM-015 | test_gpu_budget_check_ok, test_gpu_budget_exceeded |
| NFR-DREAM-003 | test_wake_latency_success (p99 < 100ms) |
| NFR-DREAM-004 | test_gpu_budget_exceeded |
| NFR-DREAM-005 | test_signal_wake, test_reset |
