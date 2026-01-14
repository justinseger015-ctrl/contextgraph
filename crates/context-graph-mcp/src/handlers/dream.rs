//! Dream Consolidation MCP Handlers
//!
//! TASK-DREAM-MCP: MCP tool handlers for dream consolidation system.
//! TASK-37: Added get_gpu_status for GPU utilization monitoring.
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.
//!
//! ## Constitution Reference
//!
//! Dream trigger conditions (constitution.yaml:446):
//! - Activity level < 0.15 for 10 minutes
//! - No active queries
//! - GPU usage < 30%
//! - Wake latency < 100ms (MANDATE)
//!
//! GPU thresholds (constitution.yaml):
//! - dream.trigger.gpu = "<80%" - Eligibility to START dream
//! - dream.constraints.gpu = "<30%" - Budget during dream (abort if exceeded)
//!
//! ## Tools
//!
//! - trigger_dream: Manually trigger dream consolidation cycle
//! - get_dream_status: Get current dream system status
//! - abort_dream: Abort current dream cycle (<100ms mandate)
//! - get_amortized_shortcuts: Get shortcut candidates from amortized learning
//! - get_gpu_status: Get GPU utilization and dream eligibility (TASK-37)

use context_graph_core::dream::DreamPhase;
use serde_json::json;
use tracing::{debug, error, info, warn};

use crate::protocol::{error_codes, JsonRpcId, JsonRpcResponse};

use super::Handlers;

impl Handlers {
    /// trigger_dream tool implementation.
    ///
    /// TASK-35: Wires MCP trigger_dream to TriggerManager.request_manual_trigger().
    /// TASK-DREAM-PH-002/003: Now accepts phase parameter for targeted dream execution.
    /// FAIL FAST if TriggerManager not initialized.
    ///
    /// Arguments:
    /// - rationale: string - REQUIRED reason for manual trigger (for audit logging)
    /// - phase: string - Optional phase (nrem/rem/full_cycle, default: full_cycle)
    /// - force: bool - Force trigger even if GPU busy (not recommended, violates constitution)
    ///
    /// Returns:
    /// - triggered: bool - Whether manual trigger was accepted
    /// - trigger_reason: string - "Manual" if accepted
    /// - gpu_utilization: Option<f32> - Current GPU usage if available
    /// - gpu_eligible: bool - Whether GPU < 80% (constitution: dream.trigger.gpu)
    /// - error: Option<string> - Error message if failed
    ///
    /// # Constitution Compliance
    /// - GPU eligibility: dream.trigger.gpu = "<80%"
    /// - GPU budget during dream: dream.constraints.gpu = "<30%"
    /// - AP-26: No silent failures - FAIL FAST on missing components
    pub(super) async fn call_trigger_dream(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling trigger_dream tool call");

        // FAIL FAST: TriggerManager is REQUIRED
        let trigger_manager = match &self.trigger_manager {
            Some(tm) => tm,
            None => {
                error!("trigger_dream: TriggerManager not initialized - FAIL FAST per AP-26");
                return JsonRpcResponse::error(
                    id,
                    error_codes::DREAM_NOT_INITIALIZED,
                    "TriggerManager not initialized. Configure with with_trigger_manager().",
                );
            }
        };

        // Parse rationale - REQUIRED for audit logging
        let rationale = match args.get("rationale").and_then(|v| v.as_str()) {
            Some(r) if !r.trim().is_empty() => r.to_string(),
            _ => {
                warn!("trigger_dream: rationale is required for audit compliance");
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "rationale is required for manual dream trigger (audit compliance)",
                );
            }
        };

        // Parse force parameter
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

        // TASK-DREAM-PH-002/003: Parse phase parameter (default: full_cycle per PRD)
        let phase = match args.get("phase").and_then(|v| v.as_str()) {
            Some("nrem") => DreamPhase::Nrem,
            Some("rem") => DreamPhase::Rem,
            Some("full_cycle") | None => DreamPhase::FullCycle,
            Some(invalid) => {
                warn!("trigger_dream: invalid phase '{}', must be nrem/rem/full_cycle", invalid);
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid phase '{}'. Must be 'nrem', 'rem', or 'full_cycle'", invalid),
                );
            }
        };

        // Check GPU eligibility (unless forced)
        // Constitution: dream.trigger.gpu = "<80%"
        let (gpu_utilization, gpu_eligible) = {
            let manager = trigger_manager.read();
            let gpu_usage = manager.current_gpu_usage();
            // Eligibility threshold is 80% per constitution
            let eligible = gpu_usage < 0.80;
            (gpu_usage, eligible)
        };

        if !gpu_eligible && !force {
            info!(
                "trigger_dream: GPU not eligible ({}% >= 80%), rationale: {}",
                gpu_utilization * 100.0,
                rationale
            );
            return self.tool_result_with_pulse(
                id,
                json!({
                    "triggered": false,
                    "trigger_reason": null,
                    "gpu_utilization": gpu_utilization,
                    "gpu_eligible": false,
                    "error": format!(
                        "GPU usage {}% >= 80% eligibility threshold. Use force=true to override (not recommended).",
                        (gpu_utilization * 100.0).round()
                    ),
                    "rationale_received": rationale,
                    "constitution_ref": "dream.trigger.gpu = '<80%'"
                }),
            );
        }

        // Check cooldown
        let cooldown_remaining = {
            let manager = trigger_manager.read();
            manager.cooldown_remaining()
        };

        if let Some(remaining) = cooldown_remaining {
            if !force {
                info!(
                    "trigger_dream: In cooldown ({}s remaining), rationale: {}",
                    remaining.as_secs(),
                    rationale
                );
                return self.tool_result_with_pulse(
                    id,
                    json!({
                        "triggered": false,
                        "trigger_reason": null,
                        "gpu_utilization": gpu_utilization,
                        "gpu_eligible": gpu_eligible,
                        "cooldown_remaining_secs": remaining.as_secs(),
                        "error": format!(
                            "Trigger cooldown active, {}s remaining. Use force=true to override.",
                            remaining.as_secs()
                        ),
                        "rationale_received": rationale
                    }),
                );
            }
        }

        // REQUEST MANUAL TRIGGER - This is the key operation
        // TASK-DREAM-PH-003: Pass phase parameter to TriggerManager
        {
            let mut manager = trigger_manager.write();
            manager.request_manual_trigger(phase);
        }

        info!(
            "trigger_dream: Manual trigger ACCEPTED, phase: {:?}, rationale: '{}', GPU: {}%, forced: {}",
            phase,
            rationale,
            (gpu_utilization * 100.0).round(),
            force
        );

        // Verify trigger was set (Full State Verification)
        let trigger_set = {
            let manager = trigger_manager.read();
            matches!(
                manager.check_triggers(),
                Some(context_graph_core::dream::types::ExtendedTriggerReason::Manual { .. })
            )
        };

        if !trigger_set {
            error!("trigger_dream: Manual trigger was NOT set after request_manual_trigger()");
            return JsonRpcResponse::error(
                id,
                error_codes::DREAM_TRIGGER_FAILED,
                "Manual trigger request failed - check_triggers() did not return Manual",
            );
        }

        self.tool_result_with_pulse(
            id,
            json!({
                "triggered": true,
                "trigger_reason": "Manual",
                "phase": phase.to_string(),  // TASK-DREAM-PH-003: Include requested phase
                "gpu_utilization": gpu_utilization,
                "gpu_eligible": gpu_eligible,
                "rationale_logged": rationale,
                "forced": force,
                "note": "Dream cycle will be executed by background scheduler when check_triggers() is called"
            }),
        )
    }

    /// get_dream_status tool implementation.
    ///
    /// TASK-DREAM-MCP: Get current dream system status.
    /// FAIL FAST if DreamController not initialized.
    ///
    /// Returns:
    /// - state: string - Current dream state (Awake/EnteringDream/Nrem/Rem/Waking)
    /// - is_dreaming: bool - Whether currently in dream cycle
    /// - scheduler: object - Scheduler state (activity, cooldown, etc.)
    /// - constitution_compliance: object - Compliance with constitution mandates
    pub(super) async fn call_get_dream_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling get_dream_status tool call");

        // FAIL FAST: Check dream controller
        let dream_controller = match &self.dream_controller {
            Some(dc) => dc,
            None => {
                error!("get_dream_status: DreamController not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::DREAM_NOT_INITIALIZED,
                    "DreamController not initialized - use with_dream() constructor",
                );
            }
        };

        // FAIL FAST: Check dream scheduler
        let dream_scheduler = match &self.dream_scheduler {
            Some(ds) => ds,
            None => {
                error!("get_dream_status: DreamScheduler not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::DREAM_NOT_INITIALIZED,
                    "DreamScheduler not initialized - use with_dream() constructor",
                );
            }
        };

        // Get controller status
        let status = {
            let controller = dream_controller.read();
            controller.get_status()
        };

        // Get scheduler info
        let scheduler_info = {
            let scheduler = dream_scheduler.read();
            json!({
                "average_activity": scheduler.get_average_activity(),
                "activity_threshold": scheduler.activity_threshold(),
                "idle_duration_trigger_secs": scheduler.idle_duration_trigger().as_secs(),
                "cooldown_remaining_secs": scheduler.cooldown_remaining().map(|d| d.as_secs()),
                "current_idle_duration_secs": scheduler.current_idle_duration().map(|d| d.as_secs()),
                "trigger_decision": format!("{:?}", scheduler.check_trigger())
            })
        };

        // Constitution compliance checks
        let constitution_compliance = json!({
            "gpu_under_30_percent": status.gpu_usage < 0.30,
            "current_gpu_usage": status.gpu_usage,
            "max_gpu_allowed": 0.30,
            "max_wake_latency_ms": 100
        });

        self.tool_result_with_pulse(
            id,
            json!({
                "state": status.state.phase_name(),
                "is_dreaming": status.is_dreaming,
                "gpu_usage": status.gpu_usage,
                "activity_level": status.activity_level,
                "completed_cycles": status.completed_cycles,
                "time_since_last_dream_secs": status.time_since_last_dream.map(|d| d.as_secs()),
                "last_dream_completed": status.last_dream_completed.map(|t| t.to_rfc3339()),
                "scheduler": scheduler_info,
                "constitution_compliance": constitution_compliance,
                "constitution_reference": {
                    "activity_threshold": 0.15,
                    "idle_duration_minutes": 10,
                    "max_wake_latency_ms": 100,
                    "max_gpu_usage": 0.30
                }
            }),
        )
    }

    /// abort_dream tool implementation.
    ///
    /// TASK-DREAM-MCP: Abort the current dream cycle.
    /// Constitution MANDATE: Must complete in <100ms.
    /// FAIL FAST if DreamController not initialized.
    ///
    /// Arguments:
    /// - reason: string - Reason for abort (optional)
    ///
    /// Returns:
    /// - aborted: bool - Whether abort was executed
    /// - abort_latency_ms: u64 - Time taken to abort
    /// - previous_state: string - State before abort
    /// - mandate_met: bool - Whether <100ms mandate was met
    pub(super) async fn call_abort_dream(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling abort_dream tool call");

        // FAIL FAST: Check dream controller
        let dream_controller = match &self.dream_controller {
            Some(dc) => dc,
            None => {
                error!("abort_dream: DreamController not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::DREAM_NOT_INITIALIZED,
                    "DreamController not initialized - use with_dream() constructor",
                );
            }
        };

        // Get previous state
        let previous_state = {
            let controller = dream_controller.read();
            controller.get_status().state.phase_name().to_string()
        };

        // Check if actually dreaming
        let is_dreaming = {
            let controller = dream_controller.read();
            controller.get_status().is_dreaming
        };

        if !is_dreaming {
            return self.tool_result_with_pulse(
                id,
                json!({
                    "aborted": false,
                    "abort_latency_ms": 0,
                    "previous_state": previous_state,
                    "mandate_met": true,
                    "reason": "Not currently dreaming - nothing to abort"
                }),
            );
        }

        // Parse optional reason
        let reason = args
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("Manual abort requested");

        // Execute abort
        let abort_result = {
            let mut controller = dream_controller.write();
            controller.abort()
        };

        match abort_result {
            Ok(wake_latency) => {
                let mandate_met = wake_latency.as_millis() < 100;

                if !mandate_met {
                    warn!(
                        "abort_dream: Constitution mandate violated - abort took {}ms (max 100ms)",
                        wake_latency.as_millis()
                    );
                }

                info!(
                    "Dream cycle aborted successfully in {}ms (reason: {})",
                    wake_latency.as_millis(),
                    reason
                );

                // Record completion in scheduler
                if let Some(scheduler) = &self.dream_scheduler {
                    let mut scheduler = scheduler.write();
                    scheduler.record_dream_completion();
                }

                self.tool_result_with_pulse(
                    id,
                    json!({
                        "aborted": true,
                        "abort_latency_ms": wake_latency.as_millis(),
                        "previous_state": previous_state,
                        "mandate_met": mandate_met,
                        "reason": reason
                    }),
                )
            }
            Err(e) => {
                error!(error = %e, "abort_dream: Failed to abort dream cycle");
                JsonRpcResponse::error(
                    id,
                    error_codes::DREAM_ABORT_ERROR,
                    format!("Failed to abort dream cycle: {}", e),
                )
            }
        }
    }

    /// get_amortized_shortcuts tool implementation.
    ///
    /// TASK-DREAM-MCP: Get shortcut candidates from amortized learning.
    /// Per constitution: Creates shortcuts for 3+ hop paths traversed 5+ times.
    /// FAIL FAST if AmortizedLearner not initialized.
    ///
    /// Arguments:
    /// - min_confidence: f32 - Minimum confidence threshold (default: 0.0)
    /// - limit: usize - Maximum number of shortcuts to return (default: 100)
    ///
    /// Returns:
    /// - shortcuts: array - List of shortcut candidates
    /// - total_candidates: usize - Total number of candidates
    /// - shortcuts_created_this_cycle: usize - Shortcuts created in current/last cycle
    pub(super) async fn call_get_amortized_shortcuts(
        &self,
        id: Option<JsonRpcId>,
        args: serde_json::Value,
    ) -> JsonRpcResponse {
        debug!("Handling get_amortized_shortcuts tool call");

        // FAIL FAST: Check amortized learner
        let amortized_learner = match &self.amortized_learner {
            Some(al) => al,
            None => {
                error!("get_amortized_shortcuts: AmortizedLearner not initialized");
                return JsonRpcResponse::error(
                    id,
                    error_codes::DREAM_NOT_INITIALIZED,
                    "AmortizedLearner not initialized - use with_dream() constructor",
                );
            }
        };

        // Parse parameters
        let min_confidence = args
            .get("min_confidence")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.0);
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(100);

        // Get candidates from amortized learner
        let (candidates, shortcuts_this_cycle) = {
            let learner = amortized_learner.read();
            let all_candidates = learner.get_candidates();
            let created = learner.shortcuts_created_this_cycle();
            (all_candidates, created)
        };

        // Filter by min_confidence and limit
        let filtered: Vec<_> = candidates
            .iter()
            .filter(|c| c.min_confidence >= min_confidence)
            .take(limit)
            .map(|c| {
                json!({
                    "source": c.source.to_string(),
                    "target": c.target.to_string(),
                    "hop_count": c.hop_count,
                    "traversal_count": c.traversal_count,
                    "combined_weight": c.combined_weight,
                    "min_confidence": c.min_confidence,
                    "path_length": c.path_nodes.len()
                })
            })
            .collect();

        self.tool_result_with_pulse(
            id,
            json!({
                "shortcuts": filtered,
                "total_candidates": candidates.len(),
                "returned_count": filtered.len(),
                "shortcuts_created_this_cycle": shortcuts_this_cycle,
                "filters_applied": {
                    "min_confidence": min_confidence,
                    "limit": limit
                },
                "constitution_reference": {
                    "min_hops": 3,
                    "min_traversals": 5
                }
            }),
        )
    }

    /// get_gpu_status tool implementation.
    ///
    /// TASK-37: Exposes GpuMonitor trait from TASK-23.
    /// FAIL FAST if GpuMonitor not initialized (AP-26).
    ///
    /// Returns:
    /// - utilization: f32 - Current GPU usage [0.0, 1.0]
    /// - is_eligible_for_dream: bool - GPU < 80% (constitution: dream.trigger.gpu)
    /// - should_abort_dream: bool - GPU > 30% (constitution: dream.constraints.gpu)
    /// - monitor_available: bool - Whether GPU monitoring is available
    /// - error: Option<string> - Error message if query failed
    ///
    /// # Constitution Compliance
    ///
    /// - dream.trigger.gpu = "<80%" - Eligibility to START dream
    /// - dream.constraints.gpu = "<30%" - Budget during dream (abort if exceeded)
    /// - AP-26: No silent failures - returns explicit error on unavailable GPU
    pub(super) async fn call_get_gpu_status(&self, id: Option<JsonRpcId>) -> JsonRpcResponse {
        debug!("Handling get_gpu_status tool call");

        // FAIL FAST: GpuMonitor is REQUIRED
        let gpu_monitor = match &self.gpu_monitor {
            Some(gm) => gm,
            None => {
                error!("get_gpu_status: GpuMonitor not initialized - FAIL FAST per AP-26");
                // Return MCP tool error (isError=true) not JSON-RPC error
                // This allows client to handle the error gracefully while still failing fast
                return self.tool_error_with_pulse(
                    id,
                    &json!({
                        "error_type": "GPU_MONITOR_NOT_INITIALIZED",
                        "message": "GpuMonitor not initialized. Configure with with_gpu_monitor() or use with_default_gwt().",
                        "error_code": error_codes::GPU_MONITOR_NOT_INITIALIZED
                    }).to_string(),
                );
            }
        };

        // Get GPU utilization
        let utilization_result = {
            let mut monitor = gpu_monitor.write();
            monitor.get_utilization()
        };

        match utilization_result {
            Ok(utilization) => {
                // Consolidate all threshold checks in a single lock acquisition for efficiency
                let (is_eligible_result, should_abort_result, monitor_available) = {
                    let mut monitor = gpu_monitor.write();
                    let eligible = monitor.is_eligible_for_dream();
                    let abort = monitor.should_abort_dream();
                    let available = monitor.is_available();
                    (eligible, abort, available)
                };

                // Handle potential errors from threshold checks - log warnings per AP-26
                // (warn on errors, use conservative defaults: not eligible, should abort)
                let is_eligible = match is_eligible_result {
                    Ok(v) => v,
                    Err(ref e) => {
                        warn!("get_gpu_status: is_eligible_for_dream() failed: {} - defaulting to false", e);
                        false
                    }
                };
                let should_abort = match should_abort_result {
                    Ok(v) => v,
                    Err(ref e) => {
                        warn!("get_gpu_status: should_abort_dream() failed: {} - defaulting to true", e);
                        true
                    }
                };

                info!(
                    "get_gpu_status: utilization={:.1}%, eligible={}, should_abort={}",
                    utilization * 100.0,
                    is_eligible,
                    should_abort
                );

                self.tool_result_with_pulse(
                    id,
                    json!({
                        "utilization": utilization,
                        "utilization_percent": format!("{:.1}%", utilization * 100.0),
                        "is_eligible_for_dream": is_eligible,
                        "should_abort_dream": should_abort,
                        "monitor_available": monitor_available,
                        "thresholds": {
                            "eligibility": {
                                "threshold": 0.80,
                                "threshold_percent": "80%",
                                "description": "GPU < 80% to START dream",
                                "constitution_ref": "dream.trigger.gpu = '<80%'"
                            },
                            "budget": {
                                "threshold": 0.30,
                                "threshold_percent": "30%",
                                "description": "GPU > 30% must ABORT dream",
                                "constitution_ref": "dream.constraints.gpu = '<30%'"
                            }
                        }
                    }),
                )
            }
            Err(e) => {
                // Per AP-26: Return explicit error, not silent 0.0
                warn!("get_gpu_status: GPU query failed: {}", e);

                // Return partial response with error details
                self.tool_result_with_pulse(
                    id,
                    json!({
                        "utilization": null,
                        "utilization_percent": null,
                        "is_eligible_for_dream": null,
                        "should_abort_dream": null,
                        "monitor_available": false,
                        "error": e.to_string(),
                        "error_type": match e {
                            context_graph_core::dream::GpuMonitorError::NvmlNotAvailable =>
                                "nvml_not_available",
                            context_graph_core::dream::GpuMonitorError::NoDevices =>
                                "no_devices",
                            context_graph_core::dream::GpuMonitorError::NvmlInitFailed(_) =>
                                "nvml_init_failed",
                            context_graph_core::dream::GpuMonitorError::DeviceAccessFailed { .. } =>
                                "device_access_failed",
                            context_graph_core::dream::GpuMonitorError::UtilizationQueryFailed(_) =>
                                "utilization_query_failed",
                            context_graph_core::dream::GpuMonitorError::Disabled =>
                                "disabled",
                        },
                        "thresholds": {
                            "eligibility": {
                                "threshold": 0.80,
                                "threshold_percent": "80%",
                                "constitution_ref": "dream.trigger.gpu = '<80%'"
                            },
                            "budget": {
                                "threshold": 0.30,
                                "threshold_percent": "30%",
                                "constitution_ref": "dream.constraints.gpu = '<30%'"
                            }
                        }
                    }),
                )
            }
        }
    }
}
