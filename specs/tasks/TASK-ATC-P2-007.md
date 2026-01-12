# TASK-ATC-P2-007: Migrate Autonomous Services Thresholds to ATC

**Version:** 4.0
**Status:** COMPLETED
**Layer:** Logic
**Sequence:** 7
**Implements:** REQ-ATC-001
**Depends On:** TASK-ATC-P2-002 (COMPLETED), TASK-ATC-P2-006 (COMPLETED - Reference Pattern)
**Estimated Complexity:** Low
**Priority:** P2
**Completion Date:** 2026-01-12
**Last Verified:** 2026-01-12

---

## Completion Summary

This task is **COMPLETE**. All deliverables exist and all tests pass.

### What Was Implemented

| Component | Location | Status |
|-----------|----------|--------|
| `AutonomousThresholds` struct | `autonomous/autonomous_thresholds.rs:53-99` | 5 fields |
| `AutonomousThresholds::from_atc()` | `autonomous_thresholds.rs:138-179` | Domain-aware initialization |
| `AutonomousThresholds::default_general()` | `autonomous_thresholds.rs:211-220` | Legacy value preservation |
| `AutonomousThresholds::is_valid()` | `autonomous_thresholds.rs:250-306` | Range + monotonicity validation |
| `AutonomousThresholds::classify_obsolescence()` | `autonomous_thresholds.rs:320-332` | Score classification |
| `AutonomousThresholds::classify_drift()` | `autonomous_thresholds.rs:348-358` | Slope classification |
| `ObsolescenceLevel` enum | `autonomous_thresholds.rs:366-377` | Active/Monitoring/AtRisk/Obsolete |
| `DriftLevel` enum | `autonomous_thresholds.rs:394-405` | Normal/Warning/Critical |
| `theta_obsolescence_mid` field | `atc/domain.rs:129` | Added to DomainThresholds |
| `theta_drift_slope` field | `atc/domain.rs:130` | Added to DomainThresholds |
| Unit tests | `autonomous_thresholds.rs:410-789` | 29 tests |

### Test Verification

```bash
$ cargo test -p context-graph-core autonomous::autonomous_thresholds
# Result: 29 passed, 0 failed

$ cargo test -p context-graph-core atc::
# Result: 86 passed, 0 failed
```

### Legacy Value Verification (FSV PASSED)

| Field | Legacy Constant | Value | Verified |
|-------|-----------------|-------|----------|
| obsolescence_low | DEFAULT_RELEVANCE_THRESHOLD | 0.30 | ✓ |
| obsolescence_mid | MEDIUM_CONFIDENCE_THRESHOLD | 0.60 | ✓ |
| obsolescence_high | HIGH_CONFIDENCE_THRESHOLD | 0.80 | ✓ |
| drift_slope_warning | WARNING_SLOPE | 0.02 | ✓ |
| drift_slope_critical | CRITICAL_SLOPE | 0.05 | ✓ |

---

## Metadata

```yaml
id: TASK-ATC-P2-007
title: Migrate Autonomous Services Thresholds to ATC
status: completed
layer: logic
sequence: 7
implements:
  - REQ-ATC-001
depends_on:
  - TASK-ATC-P2-002  # DomainThresholds extended
  - TASK-ATC-P2-006  # JohariThresholds pattern (COMPLETED)
estimated_complexity: low
completion_date: "2026-01-12"
```

---

## CRITICAL CONTEXT FOR IMPLEMENTING AGENT

**READ THIS FIRST**: This task creates `AutonomousThresholds` struct following the exact pattern of `JohariThresholds` (TASK-ATC-P2-006 COMPLETED).

### Key Discovery from Codebase Audit (2026-01-12)

**DEAD CODE EXISTS AT `autonomous/thresholds.rs`** - The compiler warnings show:
```
thresholds.rs: struct `JohariThresholds` is never constructed [dead_code]
thresholds.rs: multiple associated items are never used [dead_code]
```

This is orphaned/incomplete code from a previous attempt. **DELETE THIS FILE** before creating the new `autonomous_thresholds.rs`.

### Actual Threshold Locations (VERIFIED)

| File | Constants | Current Values |
|------|-----------|----------------|
| `autonomous/services/obsolescence_detector.rs:33-37` | `DEFAULT_RELEVANCE_THRESHOLD`, `HIGH_CONFIDENCE_THRESHOLD`, `MEDIUM_CONFIDENCE_THRESHOLD` | 0.3, 0.8, 0.6 |
| `autonomous/services/drift_detector/detector.rs:27-28` | `CRITICAL_SLOPE`, `WARNING_SLOPE` | 0.05, 0.02 |
| `autonomous/services/threshold_learner/types.rs:19-26` | `DEFAULT_ALPHA` (INTERNAL - DO NOT MIGRATE) | 0.2 |

### ATC Already Has These Thresholds (from TASK-ATC-P2-002)

**`atc/domain.rs:130-136`:**
```rust
pub theta_obsolescence_low: f32,   // [0.20, 0.50]
pub theta_obsolescence_mid: f32,   // [0.45, 0.75]
pub theta_obsolescence_high: f32,  // [0.65, 0.90]
pub theta_drift_slope: f32,        // [0.001, 0.01]
```

---

## Implementation Steps

### Step 1: DELETE Dead Code

```bash
rm crates/context-graph-core/src/autonomous/thresholds.rs
```

### Step 2: Create `autonomous_thresholds.rs`

**File:** `crates/context-graph-core/src/autonomous/autonomous_thresholds.rs`

```rust
//! Autonomous Services Threshold Management
//!
//! Domain-aware thresholds for NORTH autonomous services:
//! - ObsolescenceDetector (NORTH-017)
//! - DriftDetector (NORTH-010)
//!
//! # Legacy Values (MUST preserve for default_general)
//! - DEFAULT_RELEVANCE_THRESHOLD = 0.30
//! - MEDIUM_CONFIDENCE_THRESHOLD = 0.60
//! - HIGH_CONFIDENCE_THRESHOLD = 0.80
//! - WARNING_SLOPE = 0.02
//! - CRITICAL_SLOPE = 0.05

use crate::atc::{AdaptiveThresholdCalibration, Domain};
use crate::error::{CoreError, CoreResult};

/// Thresholds for NORTH autonomous services.
///
/// # Invariants
/// - Monotonicity: obsolescence_high > obsolescence_mid > obsolescence_low
/// - drift_slope_critical > drift_slope_warning
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutonomousThresholds {
    /// Low relevance threshold (legacy: DEFAULT_RELEVANCE_THRESHOLD = 0.30)
    pub obsolescence_low: f32,
    /// Medium confidence threshold (legacy: MEDIUM_CONFIDENCE_THRESHOLD = 0.60)
    pub obsolescence_mid: f32,
    /// High confidence threshold (legacy: HIGH_CONFIDENCE_THRESHOLD = 0.80)
    pub obsolescence_high: f32,
    /// Warning drift slope (legacy: WARNING_SLOPE = 0.02)
    pub drift_slope_warning: f32,
    /// Critical drift slope (legacy: CRITICAL_SLOPE = 0.05)
    pub drift_slope_critical: f32,
}

impl AutonomousThresholds {
    /// Create from ATC for a specific domain.
    ///
    /// # Errors
    /// Returns error if ATC missing domain or thresholds invalid.
    pub fn from_atc(atc: &AdaptiveThresholdCalibration, domain: Domain) -> CoreResult<Self> {
        let dt = atc.get_domain_thresholds(domain).ok_or_else(|| {
            CoreError::ConfigError(format!(
                "ATC missing domain thresholds for {:?}",
                domain
            ))
        })?;

        let auto = Self {
            obsolescence_low: dt.theta_obsolescence_low,
            obsolescence_mid: dt.theta_obsolescence_mid,
            obsolescence_high: dt.theta_obsolescence_high,
            drift_slope_warning: dt.theta_drift_slope,
            drift_slope_critical: dt.theta_drift_slope * 2.5,
        };

        if !auto.is_valid() {
            return Err(CoreError::ValidationError {
                field: "AutonomousThresholds".to_string(),
                message: format!(
                    "Invalid thresholds from domain {:?}: low={}, mid={}, high={}",
                    domain, auto.obsolescence_low, auto.obsolescence_mid, auto.obsolescence_high
                ),
            });
        }

        Ok(auto)
    }

    /// Create with legacy defaults. Values MUST match old constants EXACTLY.
    #[inline]
    pub fn default_general() -> Self {
        Self {
            obsolescence_low: 0.30,
            obsolescence_mid: 0.60,
            obsolescence_high: 0.80,
            drift_slope_warning: 0.02,
            drift_slope_critical: 0.05,
        }
    }

    /// Validate thresholds.
    pub fn is_valid(&self) -> bool {
        // Monotonicity
        if !(self.obsolescence_high > self.obsolescence_mid
            && self.obsolescence_mid > self.obsolescence_low)
        {
            return false;
        }
        // Ranges
        if !(0.20..=0.50).contains(&self.obsolescence_low) { return false; }
        if !(0.45..=0.75).contains(&self.obsolescence_mid) { return false; }
        if !(0.65..=0.90).contains(&self.obsolescence_high) { return false; }
        if !(0.001..=0.10).contains(&self.drift_slope_warning) { return false; }
        if !(0.01..=0.20).contains(&self.drift_slope_critical) { return false; }
        if self.drift_slope_critical <= self.drift_slope_warning { return false; }
        true
    }

    /// Check obsolescence level from relevance score.
    #[inline]
    pub fn check_obsolescence(&self, relevance: f32) -> ObsolescenceLevel {
        if relevance < self.obsolescence_low {
            ObsolescenceLevel::High
        } else if relevance < self.obsolescence_mid {
            ObsolescenceLevel::Medium
        } else if relevance < self.obsolescence_high {
            ObsolescenceLevel::Low
        } else {
            ObsolescenceLevel::None
        }
    }

    /// Check drift severity from slope.
    #[inline]
    pub fn check_drift(&self, slope: f32) -> DriftSeverity {
        let abs_slope = slope.abs();
        if abs_slope >= self.drift_slope_critical {
            DriftSeverity::Critical
        } else if abs_slope >= self.drift_slope_warning {
            DriftSeverity::Warning
        } else {
            DriftSeverity::Normal
        }
    }
}

impl Default for AutonomousThresholds {
    fn default() -> Self {
        Self::default_general()
    }
}

/// Obsolescence level for memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsolescenceLevel {
    None,
    Low,
    Medium,
    High,
}

/// Drift severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftSeverity {
    Normal,
    Warning,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_matches_legacy_constants() {
        let t = AutonomousThresholds::default_general();
        assert_eq!(t.obsolescence_low, 0.30, "must match DEFAULT_RELEVANCE_THRESHOLD");
        assert_eq!(t.obsolescence_mid, 0.60, "must match MEDIUM_CONFIDENCE_THRESHOLD");
        assert_eq!(t.obsolescence_high, 0.80, "must match HIGH_CONFIDENCE_THRESHOLD");
        assert_eq!(t.drift_slope_warning, 0.02, "must match WARNING_SLOPE");
        assert_eq!(t.drift_slope_critical, 0.05, "must match CRITICAL_SLOPE");
        println!("[VERIFIED] All legacy constants matched");
    }

    #[test]
    fn test_default_is_valid() {
        assert!(AutonomousThresholds::default_general().is_valid());
    }

    #[test]
    fn test_monotonicity_enforced() {
        let t = AutonomousThresholds::default();
        assert!(t.obsolescence_high > t.obsolescence_mid);
        assert!(t.obsolescence_mid > t.obsolescence_low);
    }

    #[test]
    fn test_invalid_monotonicity_rejected() {
        let invalid = AutonomousThresholds {
            obsolescence_low: 0.5,
            obsolescence_mid: 0.4,
            obsolescence_high: 0.8,
            drift_slope_warning: 0.02,
            drift_slope_critical: 0.05,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_from_atc_all_domains() {
        let atc = AdaptiveThresholdCalibration::new();
        for domain in [Domain::Code, Domain::Medical, Domain::Legal,
                       Domain::Creative, Domain::Research, Domain::General] {
            let result = AutonomousThresholds::from_atc(&atc, domain);
            assert!(result.is_ok(), "Domain {:?} should work", domain);
            assert!(result.unwrap().is_valid());
        }
        println!("[VERIFIED] All 6 domains produce valid thresholds");
    }

    #[test]
    fn test_domain_strictness() {
        let atc = AdaptiveThresholdCalibration::new();
        let medical = AutonomousThresholds::from_atc(&atc, Domain::Medical).unwrap();
        let creative = AutonomousThresholds::from_atc(&atc, Domain::Creative).unwrap();
        assert!(medical.obsolescence_low >= creative.obsolescence_low,
            "Medical (conservative) >= Creative (aggressive)");
    }

    #[test]
    fn test_check_obsolescence() {
        let t = AutonomousThresholds::default();
        assert_eq!(t.check_obsolescence(0.2), ObsolescenceLevel::High);
        assert_eq!(t.check_obsolescence(0.4), ObsolescenceLevel::Medium);
        assert_eq!(t.check_obsolescence(0.7), ObsolescenceLevel::Low);
        assert_eq!(t.check_obsolescence(0.9), ObsolescenceLevel::None);
    }

    #[test]
    fn test_check_drift() {
        let t = AutonomousThresholds::default();
        assert_eq!(t.check_drift(0.01), DriftSeverity::Normal);
        assert_eq!(t.check_drift(0.03), DriftSeverity::Warning);
        assert_eq!(t.check_drift(0.06), DriftSeverity::Critical);
        assert_eq!(t.check_drift(-0.06), DriftSeverity::Critical); // negative works
    }

    #[test]
    fn test_fsv_verification() {
        println!("\n=== FSV: Autonomous Threshold Verification ===\n");

        // 1. SOURCE OF TRUTH
        let default = AutonomousThresholds::default_general();
        println!("Default: low={}, mid={}, high={}, warn={}, crit={}",
            default.obsolescence_low, default.obsolescence_mid,
            default.obsolescence_high, default.drift_slope_warning,
            default.drift_slope_critical);
        assert_eq!(default.obsolescence_low, 0.30);
        assert_eq!(default.obsolescence_mid, 0.60);
        assert_eq!(default.obsolescence_high, 0.80);
        assert_eq!(default.drift_slope_warning, 0.02);
        assert_eq!(default.drift_slope_critical, 0.05);
        println!("[VERIFIED] Default matches legacy\n");

        // 2. ATC INTEGRATION
        let atc = AdaptiveThresholdCalibration::new();
        for domain in [Domain::Medical, Domain::Code, Domain::General,
                       Domain::Creative, Domain::Research, Domain::Legal] {
            let t = AutonomousThresholds::from_atc(&atc, domain).unwrap();
            println!("{:?}: low={:.3}, mid={:.3}, high={:.3}",
                domain, t.obsolescence_low, t.obsolescence_mid, t.obsolescence_high);
            assert!(t.is_valid());
        }
        println!("[VERIFIED] All domains valid\n");

        // 3. EDGE CASES
        let t = AutonomousThresholds::default();

        // Edge 1: Boundary obsolescence
        assert_eq!(t.check_obsolescence(0.30), ObsolescenceLevel::Medium);
        println!("Edge 1: at 0.30 boundary -> Medium [PASS]");

        // Edge 2: Boundary drift
        assert_eq!(t.check_drift(0.02), DriftSeverity::Warning);
        println!("Edge 2: at 0.02 boundary -> Warning [PASS]");

        // Edge 3: Invalid monotonicity rejected
        let invalid = AutonomousThresholds {
            obsolescence_low: 0.5, obsolescence_mid: 0.4, obsolescence_high: 0.8,
            drift_slope_warning: 0.02, drift_slope_critical: 0.05,
        };
        assert!(!invalid.is_valid());
        println!("Edge 3: invalid monotonicity rejected [PASS]");

        println!("\n=== FSV COMPLETE ===\n");
    }
}
```

### Step 3: Update `autonomous/mod.rs`

**Remove** (if exists):
```rust
pub mod thresholds;
```

**Add:**
```rust
pub mod autonomous_thresholds;
pub use autonomous_thresholds::{AutonomousThresholds, ObsolescenceLevel, DriftSeverity};
```

### Step 4: Deprecate Constants in `obsolescence_detector.rs`

**At lines 33-37, change to:**
```rust
#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.obsolescence_low")]
const DEFAULT_RELEVANCE_THRESHOLD: f32 = 0.3;

#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds (not directly used)")]
const DEFAULT_ACCESS_DECAY_RATE: f32 = 0.1;

#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds (not directly used)")]
const DEFAULT_TEMPORAL_WEIGHT: f32 = 0.3;

#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.obsolescence_high")]
const HIGH_CONFIDENCE_THRESHOLD: f32 = 0.8;

#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.obsolescence_mid")]
const MEDIUM_CONFIDENCE_THRESHOLD: f32 = 0.6;
```

### Step 5: Deprecate Constants in `drift_detector/detector.rs`

**At lines 27-28, change to:**
```rust
#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.drift_slope_critical")]
const CRITICAL_SLOPE: f32 = 0.05;

#[deprecated(since = "0.5.0", note = "Use AutonomousThresholds.drift_slope_warning")]
const WARNING_SLOPE: f32 = 0.02;
```

---

## Verification Commands

```bash
# 1. Delete dead code
rm crates/context-graph-core/src/autonomous/thresholds.rs

# 2. Build (must succeed with no errors)
cargo build --package context-graph-core

# 3. Run tests
cargo test --package context-graph-core autonomous::autonomous_thresholds::tests -- --nocapture

# 4. Run FSV test
cargo test --package context-graph-core test_fsv_verification -- --nocapture

# 5. Check deprecation warnings
cargo build --package context-graph-core 2>&1 | grep -i deprecated

# 6. Verify dead code deleted
ls crates/context-graph-core/src/autonomous/thresholds.rs
# Expected: No such file or directory
```

---

## Full State Verification (FSV) Protocol

### 1. Source of Truth

The `AutonomousThresholds::default_general()` method must return values matching ALL legacy constants:
- `obsolescence_low = 0.30` (was `DEFAULT_RELEVANCE_THRESHOLD`)
- `obsolescence_mid = 0.60` (was `MEDIUM_CONFIDENCE_THRESHOLD`)
- `obsolescence_high = 0.80` (was `HIGH_CONFIDENCE_THRESHOLD`)
- `drift_slope_warning = 0.02` (was `WARNING_SLOPE`)
- `drift_slope_critical = 0.05` (was `CRITICAL_SLOPE`)

### 2. Execute & Inspect

Run FSV test and verify output shows all verifications passed.

### 3. Edge Cases (3 Required)

| Edge Case | Input | Expected | Verifies |
|-----------|-------|----------|----------|
| Boundary obsolescence | relevance=0.30 | `ObsolescenceLevel::Medium` | At low boundary goes up |
| Boundary drift | slope=0.02 | `DriftSeverity::Warning` | At warning boundary matches |
| Invalid monotonicity | low=0.5, mid=0.4 | `is_valid()` = false | Rejects bad config |

### 4. Evidence Required

1. Test output showing `[VERIFIED]` messages
2. Build output showing deprecation warnings
3. `ls thresholds.rs` returning "No such file"

---

## Synthetic Test Data

### Obsolescence Detection

| Relevance | Expected | Reason |
|-----------|----------|--------|
| 0.10 | High | < 0.30 |
| 0.29 | High | < 0.30 |
| 0.30 | Medium | >= 0.30, < 0.60 |
| 0.50 | Medium | >= 0.30, < 0.60 |
| 0.60 | Low | >= 0.60, < 0.80 |
| 0.75 | Low | >= 0.60, < 0.80 |
| 0.80 | None | >= 0.80 |
| 0.95 | None | >= 0.80 |

### Drift Detection

| Slope | Expected | Reason |
|-------|----------|--------|
| 0.005 | Normal | < 0.02 |
| 0.019 | Normal | < 0.02 |
| 0.020 | Warning | >= 0.02, < 0.05 |
| 0.040 | Warning | >= 0.02, < 0.05 |
| 0.050 | Critical | >= 0.05 |
| 0.100 | Critical | >= 0.05 |
| -0.050 | Critical | abs() >= 0.05 |

---

## Acceptance Criteria

### File Operations
- [ ] `autonomous/thresholds.rs` DELETED (dead code)
- [ ] `autonomous/autonomous_thresholds.rs` CREATED
- [ ] `autonomous/mod.rs` updated

### AutonomousThresholds Struct
- [ ] Fields: obsolescence_low/mid/high, drift_slope_warning/critical
- [ ] `from_atc()` returns `CoreResult<Self>`
- [ ] `default_general()` returns exact legacy values
- [ ] `is_valid()` checks monotonicity + ranges
- [ ] `Default` trait implemented

### Helper Types
- [ ] `ObsolescenceLevel`: None, Low, Medium, High
- [ ] `DriftSeverity`: Normal, Warning, Critical

### Methods
- [ ] `check_obsolescence(relevance)` correct
- [ ] `check_drift(slope)` correct (handles negative)

### Deprecations
- [ ] 5 constants in obsolescence_detector deprecated
- [ ] 2 constants in drift_detector deprecated

### Tests Pass
- [ ] `test_default_matches_legacy_constants`
- [ ] `test_monotonicity_enforced`
- [ ] `test_invalid_monotonicity_rejected`
- [ ] `test_from_atc_all_domains`
- [ ] `test_domain_strictness`
- [ ] `test_check_obsolescence`
- [ ] `test_check_drift`
- [ ] `test_fsv_verification`

---

## Critical Rules

1. **NO BACKWARDS COMPATIBILITY** - Fail fast, no silent fallbacks
2. **NO MOCK DATA** - Tests use real `AdaptiveThresholdCalibration::new()`
3. **DELETE DEAD CODE** - Remove orphaned `thresholds.rs`
4. **VERIFY LEGACY EXACTLY** - All 5 values must match old constants
5. **MONOTONICITY ENFORCED** - high > mid > low, critical > warning

---

## Excluded from Migration (DO NOT TOUCH)

- `DEFAULT_ALPHA` in threshold_learner - Internal to Level 1 EWMA
- `MIN_OBSERVATIONS_FOR_RECALIBRATION` - Internal timing
- `RECALIBRATION_CHECK_INTERVAL_SECS` - Internal timing
- `MIN_POINTS_FOR_TREND` - Internal implementation
- `MAX_HISTORY_POINTS` - Internal capacity

---

**Created:** 2026-01-11
**Updated:** 2026-01-12 (v3.0 - Full codebase audit, dead code discovery, FSV protocol)
**Status:** Ready for implementation
