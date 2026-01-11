//! Workspace Event Listeners
//!
//! Implements listeners for workspace events that wire to subsystems:
//! - DreamEventListener: Queues exiting memories for dream replay
//! - NeuromodulationEventListener: Boosts dopamine on memory entry
//! - MetaCognitiveEventListener: Triggers epistemic action on workspace empty
//!
//! ## Constitution Reference
//!
//! From constitution.yaml:
//! - neuromod.Dopamine.trigger: "memory_enters_workspace" (lines 162-170)
//! - gwt.global_workspace step 6: "Inhibit: losing candidates receive dopamine reduction"
//! - gwt.workspace_events: memory_exits → dream replay, workspace_empty → epistemic action

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::gwt::meta_cognitive::MetaCognitiveLoop;
use crate::gwt::workspace::{WorkspaceEvent, WorkspaceEventListener};
use crate::neuromod::NeuromodulationManager;

/// Listener that queues exiting memories for dream replay
///
/// When a memory exits the workspace (r dropped below 0.7), it is queued
/// for offline dream replay consolidation.
pub struct DreamEventListener {
    dream_queue: Arc<RwLock<Vec<Uuid>>>,
}

impl DreamEventListener {
    /// Create a new dream event listener with the given queue
    pub fn new(dream_queue: Arc<RwLock<Vec<Uuid>>>) -> Self {
        Self { dream_queue }
    }

    /// Get a clone of the dream queue arc for external access
    pub fn queue(&self) -> Arc<RwLock<Vec<Uuid>>> {
        Arc::clone(&self.dream_queue)
    }
}

impl WorkspaceEventListener for DreamEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::MemoryExits {
                id,
                order_parameter,
                timestamp: _,
            } => {
                // Queue memory for dream replay - non-blocking acquire
                match self.dream_queue.try_write() {
                    Ok(mut queue) => {
                        queue.push(*id);
                        tracing::debug!(
                            "Queued memory {:?} for dream replay (r={:.3})",
                            id,
                            order_parameter
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            "CRITICAL: Failed to acquire dream_queue lock: {:?}",
                            e
                        );
                        panic!("DreamEventListener: Lock poisoned or deadlocked");
                    }
                }
            }
            WorkspaceEvent::IdentityCritical {
                identity_coherence,
                reason,
                timestamp: _,
            } => {
                // Log identity critical - DreamController handles separately via direct wiring
                tracing::warn!(
                    "Identity critical (IC={:.3}): {}",
                    identity_coherence,
                    reason
                );
            }
            // No-op for other events
            WorkspaceEvent::MemoryEnters { .. } => {}
            WorkspaceEvent::WorkspaceConflict { .. } => {}
            WorkspaceEvent::WorkspaceEmpty { .. } => {}
        }
    }
}

impl std::fmt::Debug for DreamEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DreamEventListener").finish()
    }
}

/// Listener that boosts dopamine on memory entry
///
/// When a memory enters the workspace (r crossed 0.8 upward), dopamine
/// is increased by DA_WORKSPACE_INCREMENT (0.2).
pub struct NeuromodulationEventListener {
    neuromod_manager: Arc<RwLock<NeuromodulationManager>>,
}

impl NeuromodulationEventListener {
    /// Create a new neuromodulation event listener
    pub fn new(neuromod_manager: Arc<RwLock<NeuromodulationManager>>) -> Self {
        Self { neuromod_manager }
    }

    /// Get a reference to the neuromod manager arc
    pub fn neuromod(&self) -> Arc<RwLock<NeuromodulationManager>> {
        Arc::clone(&self.neuromod_manager)
    }
}

impl WorkspaceEventListener for NeuromodulationEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::MemoryEnters {
                id,
                order_parameter,
                timestamp: _,
            } => {
                // Boost dopamine on workspace entry - non-blocking acquire
                match self.neuromod_manager.try_write() {
                    Ok(mut mgr) => {
                        mgr.on_workspace_entry();
                        tracing::debug!(
                            "Dopamine boosted for memory {:?} entering workspace (r={:.3})",
                            id,
                            order_parameter
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            "CRITICAL: Failed to acquire neuromod_manager lock: {:?}",
                            e
                        );
                        panic!("NeuromodulationEventListener: Lock poisoned or deadlocked");
                    }
                }
            }
            // No-op for other events
            WorkspaceEvent::MemoryExits { .. } => {}
            WorkspaceEvent::WorkspaceConflict { .. } => {}
            WorkspaceEvent::WorkspaceEmpty { .. } => {}
            WorkspaceEvent::IdentityCritical { .. } => {}
        }
    }
}

impl std::fmt::Debug for NeuromodulationEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuromodulationEventListener").finish()
    }
}

/// Listener that triggers epistemic action on workspace empty
///
/// When the workspace is empty for an extended period, an epistemic action
/// flag is set to trigger exploratory behavior.
pub struct MetaCognitiveEventListener {
    meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,
    epistemic_action_triggered: Arc<AtomicBool>,
}

impl MetaCognitiveEventListener {
    /// Create a new meta-cognitive event listener
    pub fn new(
        meta_cognitive: Arc<RwLock<MetaCognitiveLoop>>,
        epistemic_action_triggered: Arc<AtomicBool>,
    ) -> Self {
        Self {
            meta_cognitive,
            epistemic_action_triggered,
        }
    }

    /// Check if epistemic action has been triggered
    pub fn is_epistemic_action_triggered(&self) -> bool {
        self.epistemic_action_triggered.load(Ordering::SeqCst)
    }

    /// Reset the epistemic action flag
    pub fn reset_epistemic_action(&self) {
        self.epistemic_action_triggered.store(false, Ordering::SeqCst);
    }

    /// Get a reference to the meta-cognitive loop arc
    pub fn meta_cognitive(&self) -> Arc<RwLock<MetaCognitiveLoop>> {
        Arc::clone(&self.meta_cognitive)
    }
}

impl WorkspaceEventListener for MetaCognitiveEventListener {
    fn on_event(&self, event: &WorkspaceEvent) {
        match event {
            WorkspaceEvent::WorkspaceEmpty {
                duration_ms,
                timestamp: _,
            } => {
                // Set epistemic action flag - atomic, no lock needed
                self.epistemic_action_triggered.store(true, Ordering::SeqCst);
                tracing::info!(
                    "Workspace empty for {}ms - epistemic action triggered",
                    duration_ms
                );
            }
            // No-op for other events
            WorkspaceEvent::MemoryEnters { .. } => {}
            WorkspaceEvent::MemoryExits { .. } => {}
            WorkspaceEvent::WorkspaceConflict { .. } => {}
            WorkspaceEvent::IdentityCritical { .. } => {}
        }
    }
}

impl std::fmt::Debug for MetaCognitiveEventListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaCognitiveEventListener")
            .field(
                "epistemic_action_triggered",
                &self.epistemic_action_triggered.load(Ordering::SeqCst),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromod::DA_BASELINE;
    use chrono::Utc;

    // ============================================================
    // FSV Tests for DreamEventListener
    // ============================================================

    #[tokio::test]
    async fn test_fsv_dream_listener_memory_exits() {
        println!("=== FSV: DreamEventListener - MemoryExits ===");

        // SETUP
        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(dream_queue.clone());
        let memory_id = Uuid::new_v4();

        // BEFORE
        let before_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        println!("BEFORE: queue.len() = {}", before_len);
        assert_eq!(before_len, 0, "Queue must start empty");

        // EXECUTE
        let event = WorkspaceEvent::MemoryExits {
            id: memory_id,
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER - SEPARATE READ
        let after_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        let queued_id = {
            let queue = dream_queue.read().await;
            queue.first().cloned()
        };
        println!("AFTER: queue.len() = {}", after_len);

        // VERIFY
        assert_eq!(after_len, 1, "Queue must have exactly 1 item");
        assert_eq!(queued_id, Some(memory_id), "Queued ID must match");

        // EVIDENCE
        println!("EVIDENCE: Memory {:?} correctly queued for dream replay", memory_id);
    }

    #[tokio::test]
    async fn test_dream_listener_ignores_other_events() {
        println!("=== TEST: DreamEventListener ignores non-MemoryExits ===");

        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(dream_queue.clone());

        // Send MemoryEnters - should be ignored
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // Send WorkspaceEmpty - should be ignored
        let event = WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 1000,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let queue_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };

        assert_eq!(queue_len, 0, "Queue should remain empty for non-MemoryExits events");
        println!("EVIDENCE: DreamEventListener correctly ignores non-MemoryExits events");
    }

    #[tokio::test]
    async fn test_dream_listener_identity_critical() {
        println!("=== TEST: DreamEventListener handles IdentityCritical ===");

        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let listener = DreamEventListener::new(dream_queue.clone());

        // Send IdentityCritical - should log but not queue
        let event = WorkspaceEvent::IdentityCritical {
            identity_coherence: 0.35,
            reason: "Test critical".to_string(),
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let queue_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };

        assert_eq!(queue_len, 0, "Queue should remain empty for IdentityCritical");
        println!("EVIDENCE: IdentityCritical event handled without queuing");
    }

    // ============================================================
    // FSV Tests for NeuromodulationEventListener
    // ============================================================

    #[tokio::test]
    async fn test_fsv_neuromod_listener_dopamine_boost() {
        println!("=== FSV: NeuromodulationEventListener - Dopamine Boost ===");

        // SETUP
        let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
        let listener = NeuromodulationEventListener::new(neuromod.clone());
        let memory_id = Uuid::new_v4();

        // BEFORE - Read via separate lock
        let before_da = {
            let mgr = neuromod.read().await;
            mgr.get_hopfield_beta() // Returns dopamine value
        };
        println!("BEFORE: dopamine = {:.3}", before_da);
        assert!(
            (before_da - DA_BASELINE).abs() < f32::EPSILON,
            "Dopamine must start at baseline"
        );

        // EXECUTE
        let event = WorkspaceEvent::MemoryEnters {
            id: memory_id,
            order_parameter: 0.85,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER - Read via SEPARATE lock
        let after_da = {
            let mgr = neuromod.read().await;
            mgr.get_hopfield_beta()
        };
        println!("AFTER: dopamine = {:.3}", after_da);

        // VERIFY
        let expected_da = before_da + 0.2; // DA_WORKSPACE_INCREMENT
        assert!(
            (after_da - expected_da).abs() < f32::EPSILON,
            "Expected dopamine {:.3}, got {:.3}",
            expected_da,
            after_da
        );

        // EVIDENCE
        println!(
            "EVIDENCE: Dopamine correctly increased by 0.2 (from {:.3} to {:.3})",
            before_da, after_da
        );
    }

    #[tokio::test]
    async fn test_neuromod_listener_ignores_other_events() {
        println!("=== TEST: NeuromodulationEventListener ignores non-MemoryEnters ===");

        let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
        let listener = NeuromodulationEventListener::new(neuromod.clone());

        let initial_da = {
            let mgr = neuromod.read().await;
            mgr.get_hopfield_beta()
        };

        // Send MemoryExits - should be ignored
        let event = WorkspaceEvent::MemoryExits {
            id: Uuid::new_v4(),
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // Send WorkspaceEmpty - should be ignored
        let event = WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 1000,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let final_da = {
            let mgr = neuromod.read().await;
            mgr.get_hopfield_beta()
        };

        assert!(
            (final_da - initial_da).abs() < f32::EPSILON,
            "Dopamine should remain unchanged for non-MemoryEnters events"
        );
        println!("EVIDENCE: NeuromodulationEventListener correctly ignores non-MemoryEnters events");
    }

    #[tokio::test]
    async fn test_neuromod_listener_at_max() {
        println!("=== EDGE CASE: NeuromodulationEventListener at max dopamine ===");

        let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
        let listener = NeuromodulationEventListener::new(neuromod.clone());

        // Set dopamine to max
        {
            let mut mgr = neuromod.write().await;
            use crate::neuromod::ModulatorType;
            mgr.set(ModulatorType::Dopamine, 5.0).unwrap();
        }

        let before_da = {
            let mgr = neuromod.read().await;
            mgr.get_hopfield_beta()
        };
        println!("BEFORE: dopamine = {:.3} (at max)", before_da);

        // Trigger event
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        let after_da = {
            let mgr = neuromod.read().await;
            mgr.get_hopfield_beta()
        };
        println!("AFTER: dopamine = {:.3}", after_da);

        // Verify clamped to max
        assert!(
            after_da <= 5.0,
            "Dopamine must be clamped to max (5.0), got {}",
            after_da
        );
        println!("EVIDENCE: Dopamine correctly clamped to max");
    }

    // ============================================================
    // FSV Tests for MetaCognitiveEventListener
    // ============================================================

    #[tokio::test]
    async fn test_fsv_meta_listener_workspace_empty() {
        println!("=== FSV: MetaCognitiveEventListener - WorkspaceEmpty ===");

        // SETUP
        let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
        let epistemic_flag = Arc::new(AtomicBool::new(false));
        let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

        // BEFORE
        let before_flag = listener.is_epistemic_action_triggered();
        println!("BEFORE: epistemic_action_triggered = {}", before_flag);
        assert!(!before_flag, "Flag must start as false");

        // EXECUTE
        let event = WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 500,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // AFTER
        let after_flag = listener.is_epistemic_action_triggered();
        println!("AFTER: epistemic_action_triggered = {}", after_flag);

        // VERIFY
        assert!(after_flag, "Flag must be set to true");

        // EVIDENCE
        println!("EVIDENCE: Epistemic action flag correctly set on WorkspaceEmpty");
    }

    #[tokio::test]
    async fn test_meta_listener_ignores_other_events() {
        println!("=== TEST: MetaCognitiveEventListener ignores non-WorkspaceEmpty ===");

        let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
        let epistemic_flag = Arc::new(AtomicBool::new(false));
        let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

        // Send MemoryEnters - should be ignored
        let event = WorkspaceEvent::MemoryEnters {
            id: Uuid::new_v4(),
            order_parameter: 0.85,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        // Send MemoryExits - should be ignored
        let event = WorkspaceEvent::MemoryExits {
            id: Uuid::new_v4(),
            order_parameter: 0.65,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            !listener.is_epistemic_action_triggered(),
            "Flag should remain false for non-WorkspaceEmpty events"
        );
        println!("EVIDENCE: MetaCognitiveEventListener correctly ignores non-WorkspaceEmpty events");
    }

    #[tokio::test]
    async fn test_meta_listener_zero_duration() {
        println!("=== EDGE CASE: MetaCognitiveEventListener with duration=0 ===");

        let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
        let epistemic_flag = Arc::new(AtomicBool::new(false));
        let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

        // Execute with zero duration
        let event = WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 0,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);

        assert!(
            listener.is_epistemic_action_triggered(),
            "Flag should be set even with zero duration"
        );
        println!("EVIDENCE: Zero duration handled correctly");
    }

    #[tokio::test]
    async fn test_meta_listener_reset() {
        println!("=== TEST: MetaCognitiveEventListener reset ===");

        let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
        let epistemic_flag = Arc::new(AtomicBool::new(false));
        let listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

        // Trigger the flag
        let event = WorkspaceEvent::WorkspaceEmpty {
            duration_ms: 100,
            timestamp: Utc::now(),
        };
        listener.on_event(&event);
        assert!(listener.is_epistemic_action_triggered());

        // Reset
        listener.reset_epistemic_action();
        assert!(!listener.is_epistemic_action_triggered());

        println!("EVIDENCE: Epistemic action flag correctly reset");
    }

    // ============================================================
    // Integration test: All listeners receive events
    // ============================================================

    #[tokio::test]
    async fn test_all_listeners_receive_all_events() {
        println!("=== INTEGRATION: All listeners receive all event types ===");

        // Setup all listeners
        let dream_queue = Arc::new(RwLock::new(Vec::new()));
        let dream_listener = DreamEventListener::new(dream_queue.clone());

        let neuromod = Arc::new(RwLock::new(NeuromodulationManager::new()));
        let neuromod_listener = NeuromodulationEventListener::new(neuromod.clone());

        let meta_cognitive = Arc::new(RwLock::new(MetaCognitiveLoop::new()));
        let epistemic_flag = Arc::new(AtomicBool::new(false));
        let meta_listener = MetaCognitiveEventListener::new(meta_cognitive.clone(), epistemic_flag.clone());

        // Create events
        let events = vec![
            WorkspaceEvent::MemoryEnters {
                id: Uuid::new_v4(),
                order_parameter: 0.85,
                timestamp: Utc::now(),
            },
            WorkspaceEvent::MemoryExits {
                id: Uuid::new_v4(),
                order_parameter: 0.65,
                timestamp: Utc::now(),
            },
            WorkspaceEvent::WorkspaceEmpty {
                duration_ms: 500,
                timestamp: Utc::now(),
            },
            WorkspaceEvent::WorkspaceConflict {
                memories: vec![Uuid::new_v4(), Uuid::new_v4()],
                timestamp: Utc::now(),
            },
            WorkspaceEvent::IdentityCritical {
                identity_coherence: 0.4,
                reason: "Test".to_string(),
                timestamp: Utc::now(),
            },
        ];

        // Broadcast to all listeners (no panics expected)
        for event in &events {
            dream_listener.on_event(event);
            neuromod_listener.on_event(event);
            meta_listener.on_event(event);
        }

        // Verify expected state changes
        let dream_queue_len = {
            let queue = dream_queue.read().await;
            queue.len()
        };
        assert_eq!(dream_queue_len, 1, "Dream queue should have 1 MemoryExits");

        let final_da = {
            let mgr = neuromod.read().await;
            mgr.get_hopfield_beta()
        };
        assert!(final_da > DA_BASELINE, "Dopamine should be above baseline");

        assert!(
            meta_listener.is_epistemic_action_triggered(),
            "Epistemic flag should be set"
        );

        println!("EVIDENCE: All listeners correctly processed all event types without panic");
    }
}
