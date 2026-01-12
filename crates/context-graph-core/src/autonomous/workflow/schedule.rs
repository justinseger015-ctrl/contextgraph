//! Daily optimization schedule types.

use serde::{Deserialize, Serialize};

/// Daily optimization schedule
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DailySchedule {
    /// Low activity window for heavy operations (start_hour, end_hour)
    pub consolidation_window: (u32, u32), // default: (0, 2) = midnight to 2am

    /// Morning drift check hour (0-23)
    pub drift_check_hour: u32, // default: 6

    /// Mid-day statistics collection hour (0-23)
    pub stats_hour: u32, // default: 12

    /// Evening preparation hour (0-23)
    pub prep_hour: u32, // default: 18
}

impl Default for DailySchedule {
    fn default() -> Self {
        Self {
            consolidation_window: (0, 2),
            drift_check_hour: 6,
            stats_hour: 12,
            prep_hour: 18,
        }
    }
}

impl DailySchedule {
    /// Check if a given hour falls within the consolidation window
    pub fn is_consolidation_hour(&self, hour: u32) -> bool {
        let (start, end) = self.consolidation_window;
        if start <= end {
            hour >= start && hour < end
        } else {
            // Handles wrap-around (e.g., 22 to 2)
            hour >= start || hour < end
        }
    }

    /// Get the next scheduled check type for a given hour
    pub fn next_check_for_hour(&self, hour: u32) -> Option<ScheduledCheckType> {
        if self.is_consolidation_hour(hour) {
            Some(ScheduledCheckType::ConsolidationWindow)
        } else if hour == self.drift_check_hour {
            Some(ScheduledCheckType::DriftCheck)
        } else if hour == self.stats_hour {
            Some(ScheduledCheckType::StatisticsCollection)
        } else if hour == self.prep_hour {
            Some(ScheduledCheckType::IndexOptimization)
        } else {
            None
        }
    }

    /// Validate that the schedule is consistent
    pub fn validate(&self) -> Result<(), ScheduleValidationError> {
        if self.drift_check_hour > 23 {
            return Err(ScheduleValidationError::InvalidHour {
                field: "drift_check_hour".into(),
                value: self.drift_check_hour,
            });
        }
        if self.stats_hour > 23 {
            return Err(ScheduleValidationError::InvalidHour {
                field: "stats_hour".into(),
                value: self.stats_hour,
            });
        }
        if self.prep_hour > 23 {
            return Err(ScheduleValidationError::InvalidHour {
                field: "prep_hour".into(),
                value: self.prep_hour,
            });
        }
        if self.consolidation_window.0 > 23 || self.consolidation_window.1 > 23 {
            return Err(ScheduleValidationError::InvalidConsolidationWindow {
                start: self.consolidation_window.0,
                end: self.consolidation_window.1,
            });
        }
        Ok(())
    }
}

/// Schedule validation error
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleValidationError {
    InvalidHour { field: String, value: u32 },
    InvalidConsolidationWindow { start: u32, end: u32 },
}

impl std::fmt::Display for ScheduleValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHour { field, value } => {
                write!(
                    f,
                    "Invalid hour {} for field '{}' (must be 0-23)",
                    value, field
                )
            }
            Self::InvalidConsolidationWindow { start, end } => {
                write!(
                    f,
                    "Invalid consolidation window ({}, {}) - hours must be 0-23",
                    start, end
                )
            }
        }
    }
}

impl std::error::Error for ScheduleValidationError {}

/// Scheduled check type for optimization events
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScheduledCheckType {
    /// Daily drift check
    DriftCheck,
    /// Consolidation window for heavy operations
    ConsolidationWindow,
    /// Statistics collection and reporting
    StatisticsCollection,
    /// Index optimization and maintenance
    IndexOptimization,
}
