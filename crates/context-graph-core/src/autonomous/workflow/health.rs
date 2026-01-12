//! Autonomous system health status types.

use serde::{Deserialize, Serialize};

/// Autonomous system health status
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum AutonomousHealth {
    /// System is operating normally
    #[default]
    Healthy,
    /// System has warnings but is operational
    Warning { message: String },
    /// System has errors
    Error { message: String, recoverable: bool },
}

impl AutonomousHealth {
    /// Create a warning status
    pub fn warning(message: impl Into<String>) -> Self {
        Self::Warning {
            message: message.into(),
        }
    }

    /// Create a recoverable error status
    pub fn recoverable_error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
            recoverable: true,
        }
    }

    /// Create a fatal error status
    pub fn fatal_error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
            recoverable: false,
        }
    }

    /// Check if the system is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Check if the system can continue operating
    pub fn can_continue(&self) -> bool {
        match self {
            Self::Healthy | Self::Warning { .. } => true,
            Self::Error { recoverable, .. } => *recoverable,
        }
    }

    /// Get the message if any
    pub fn message(&self) -> Option<&str> {
        match self {
            Self::Healthy => None,
            Self::Warning { message } | Self::Error { message, .. } => Some(message),
        }
    }
}
