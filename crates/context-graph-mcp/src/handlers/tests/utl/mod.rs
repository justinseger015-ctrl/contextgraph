//! UTL Handler Tests
//!
//! TASK-UTL-P1-001: Tests for gwt/compute_delta_sc handler.
//! Tests verify:
//! - Per-embedder ΔS computation
//! - Aggregate ΔS/ΔC values
//! - Johari quadrant classification
//! - AP-10 compliance (all values in [0,1], no NaN/Inf)
//! - FAIL FAST error handling

mod helpers;
mod delta_sc_valid;
mod delta_sc_errors;
mod basic_utl;
mod fsv;
mod edge_cases;
mod property_based;
