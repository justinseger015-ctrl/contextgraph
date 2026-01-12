//! Type definitions for the Self-Healing Manager Service (NORTH-020)
//!
//! This module contains all the core types used by the self-healing system:
//! - Configuration types
//! - Health state types
//! - Issue and severity types
//! - Healing action and result types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Health score configuration for the self-healing manager
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfHealingConfig {
    /// Health threshold below which healing is triggered
    pub health_threshold: f32,
    /// Maximum healing attempts per component per hour
    pub max_healing_attempts: u32,
    /// Enable automatic healing
    pub auto_heal: bool,
    /// Cooldown between healing actions (seconds)
    pub healing_cooldown_secs: u64,
}

impl Default for SelfHealingConfig {
    fn default() -> Self {
        Self {
            health_threshold: 0.70,
            max_healing_attempts: 3,
            auto_heal: true,
            healing_cooldown_secs: 60,
        }
    }
}

/// System health state with detailed metrics for self-healing
///
/// This is the foundation type for the self-healing manager, containing
/// overall health score, per-component scores, timestamps, and active issues.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemHealthState {
    /// Overall health score [0.0, 1.0]
    pub overall_score: f32,
    /// Per-component health scores
    pub component_scores: HashMap<String, f32>,
    /// Timestamp of last health check
    pub last_check: DateTime<Utc>,
    /// Current active issues
    pub issues: Vec<HealthIssue>,
}

impl Default for SystemHealthState {
    fn default() -> Self {
        Self {
            overall_score: 1.0,
            component_scores: HashMap::new(),
            last_check: Utc::now(),
            issues: Vec::new(),
        }
    }
}

impl SystemHealthState {
    /// Create a healthy state with no issues
    pub fn healthy() -> Self {
        Self::default()
    }

    /// Create health with specific overall score
    pub fn with_score(score: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&score),
            "Health score must be in [0.0, 1.0]"
        );
        Self {
            overall_score: score,
            component_scores: HashMap::new(),
            last_check: Utc::now(),
            issues: Vec::new(),
        }
    }

    /// Add a component score
    pub fn add_component(&mut self, name: impl Into<String>, score: f32) {
        assert!(
            (0.0..=1.0).contains(&score),
            "Component score must be in [0.0, 1.0]"
        );
        self.component_scores.insert(name.into(), score);
        self.recalculate_overall();
    }

    /// Recalculate overall score from component scores
    fn recalculate_overall(&mut self) {
        if self.component_scores.is_empty() {
            return;
        }
        let sum: f32 = self.component_scores.values().sum();
        self.overall_score = sum / self.component_scores.len() as f32;
    }

    /// Add an issue to the health state
    pub fn add_issue(&mut self, issue: HealthIssue) {
        self.issues.push(issue);
    }

    /// Check if there are any unresolved issues
    pub fn has_unresolved_issues(&self) -> bool {
        self.issues.iter().any(|i| !i.resolved)
    }

    /// Get count of unresolved issues
    pub fn unresolved_count(&self) -> usize {
        self.issues.iter().filter(|i| !i.resolved).count()
    }
}

/// System operational status for self-healing
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemOperationalStatus {
    /// System is running normally
    #[default]
    Running,
    /// System is paused
    Paused,
    /// System is in degraded state but operational
    Degraded,
    /// System has failed
    Failed,
    /// System is recovering from a failure
    Recovering,
}

/// Severity level for health issues
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational - no action needed
    Info,
    /// Warning - monitor closely
    Warning,
    /// Error - needs attention
    Error,
    /// Critical - immediate action required
    Critical,
}

impl IssueSeverity {
    /// Check if this severity requires healing action
    pub fn requires_action(&self) -> bool {
        matches!(self, Self::Error | Self::Critical)
    }
}

/// A health issue detected in the system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Component where issue was detected
    pub component: String,
    /// Severity of the issue
    pub severity: IssueSeverity,
    /// Human-readable description
    pub description: String,
    /// When the issue was detected
    pub detected_at: DateTime<Utc>,
    /// Whether the issue has been resolved
    pub resolved: bool,
}

impl HealthIssue {
    /// Create a new health issue
    pub fn new(
        component: impl Into<String>,
        severity: IssueSeverity,
        description: impl Into<String>,
    ) -> Self {
        Self {
            component: component.into(),
            severity,
            description: description.into(),
            detected_at: Utc::now(),
            resolved: false,
        }
    }

    /// Create a warning issue
    pub fn warning(component: impl Into<String>, description: impl Into<String>) -> Self {
        Self::new(component, IssueSeverity::Warning, description)
    }

    /// Create an error issue
    pub fn error(component: impl Into<String>, description: impl Into<String>) -> Self {
        Self::new(component, IssueSeverity::Error, description)
    }

    /// Create a critical issue
    pub fn critical(component: impl Into<String>, description: impl Into<String>) -> Self {
        Self::new(component, IssueSeverity::Critical, description)
    }

    /// Mark issue as resolved
    pub fn resolve(&mut self) {
        self.resolved = true;
    }
}

/// Actions that can be taken to heal the system
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealingAction {
    /// Restart a specific component
    RestartComponent { name: String },
    /// Clear cache for a component
    ClearCache { component: String },
    /// Reset state for a component
    ResetState { component: String },
    /// Escalate to external handler
    Escalate { reason: String },
    /// No action needed
    NoAction,
}

impl HealingAction {
    /// Get a description of this action
    pub fn description(&self) -> String {
        match self {
            Self::RestartComponent { name } => format!("Restart component: {}", name),
            Self::ClearCache { component } => format!("Clear cache for: {}", component),
            Self::ResetState { component } => format!("Reset state for: {}", component),
            Self::Escalate { reason } => format!("Escalate: {}", reason),
            Self::NoAction => "No action required".to_string(),
        }
    }
}

/// Result of a healing action
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealingResult {
    /// The action that was taken
    pub action_taken: HealingAction,
    /// Whether the healing was successful
    pub success: bool,
    /// Health state before healing
    pub health_before: f32,
    /// Health state after healing
    pub health_after: f32,
    /// Timestamp of the healing action
    pub timestamp: DateTime<Utc>,
    /// Optional message about the result
    pub message: Option<String>,
}

impl HealingResult {
    /// Create a successful healing result
    pub fn success(action: HealingAction, health_before: f32, health_after: f32) -> Self {
        Self {
            action_taken: action,
            success: true,
            health_before,
            health_after,
            timestamp: Utc::now(),
            message: None,
        }
    }

    /// Create a failed healing result
    pub fn failure(
        action: HealingAction,
        health_before: f32,
        health_after: f32,
        message: impl Into<String>,
    ) -> Self {
        Self {
            action_taken: action,
            success: false,
            health_before,
            health_after,
            timestamp: Utc::now(),
            message: Some(message.into()),
        }
    }

    /// Check if health improved
    pub fn health_improved(&self) -> bool {
        self.health_after > self.health_before
    }
}
