//! NT weight validation wrapper with GraphError integration.
//!
//! This module provides a Result-returning wrapper around `NeurotransmitterWeights::validate()`.
//! While the core crate returns `bool`, graph operations need `GraphResult<()>` for
//! proper error propagation via the `?` operator.
//!
//! # Constitution Reference
//! - AP-001: Never unwrap() in prod - all errors properly typed
//! - edge_model.nt_weights: All weights in [0.0, 1.0]
//!
//! # Example
//! ```rust
//! use context_graph_graph::marblestone::validate_or_error;
//! use context_graph_graph::{NeurotransmitterWeights, Domain};
//!
//! fn example() -> context_graph_graph::GraphResult<()> {
//!     let weights = NeurotransmitterWeights::for_domain(Domain::Code);
//!     validate_or_error(&weights)?;  // Uses ? operator
//!     Ok(())
//! }
//! ```

use crate::error::{GraphError, GraphResult};
use context_graph_core::marblestone::NeurotransmitterWeights;

/// Validate NT weights and return detailed error on failure.
///
/// Wraps `NeurotransmitterWeights::validate()` with `GraphError` for
/// proper error propagation in graph operations. Fails fast with
/// robust error logging on the first invalid field.
///
/// # Arguments
/// * `weights` - NT weights to validate
///
/// # Returns
/// * `Ok(())` - All weights in [0.0, 1.0]
/// * `Err(GraphError::InvalidNtWeights)` - First invalid weight with details
///
/// # Error Field Order
/// Checks in order: excitatory → inhibitory → modulatory.
/// Returns error on first failure.
///
/// # Example
/// ```rust
/// use context_graph_graph::marblestone::validate_or_error;
/// use context_graph_graph::NeurotransmitterWeights;
///
/// let valid = NeurotransmitterWeights::new(0.5, 0.3, 0.4);
/// assert!(validate_or_error(&valid).is_ok());
///
/// let invalid = NeurotransmitterWeights::new(1.5, 0.0, 0.0);
/// let err = validate_or_error(&invalid).unwrap_err();
/// // err contains field="excitatory", value=1.5
/// ```
pub fn validate_or_error(weights: &NeurotransmitterWeights) -> GraphResult<()> {
    // Check excitatory field
    if weights.excitatory < 0.0
        || weights.excitatory > 1.0
        || weights.excitatory.is_nan()
        || weights.excitatory.is_infinite()
    {
        log::error!(
            "VALIDATION FAILED: excitatory={} (must be in [0.0, 1.0])",
            weights.excitatory
        );
        return Err(GraphError::InvalidNtWeights {
            field: "excitatory".to_string(),
            value: weights.excitatory,
        });
    }

    // Check inhibitory field
    if weights.inhibitory < 0.0
        || weights.inhibitory > 1.0
        || weights.inhibitory.is_nan()
        || weights.inhibitory.is_infinite()
    {
        log::error!(
            "VALIDATION FAILED: inhibitory={} (must be in [0.0, 1.0])",
            weights.inhibitory
        );
        return Err(GraphError::InvalidNtWeights {
            field: "inhibitory".to_string(),
            value: weights.inhibitory,
        });
    }

    // Check modulatory field
    if weights.modulatory < 0.0
        || weights.modulatory > 1.0
        || weights.modulatory.is_nan()
        || weights.modulatory.is_infinite()
    {
        log::error!(
            "VALIDATION FAILED: modulatory={} (must be in [0.0, 1.0])",
            weights.modulatory
        );
        return Err(GraphError::InvalidNtWeights {
            field: "modulatory".to_string(),
            value: weights.modulatory,
        });
    }

    log::debug!(
        "Validation passed: e={}, i={}, m={}",
        weights.excitatory,
        weights.inhibitory,
        weights.modulatory
    );
    Ok(())
}

/// Validate and compute effective weight in one call.
///
/// Fails fast if weights are invalid, otherwise computes effective weight.
/// This combines validation and computation for convenience in graph operations.
///
/// # Arguments
/// * `weights` - NT weights to validate and use
/// * `base_weight` - Base edge weight [0.0, 1.0]
///
/// # Returns
/// * `Ok(f32)` - Effective weight after modulation
/// * `Err(GraphError::InvalidNtWeights)` - Validation failed
///
/// # Formula
/// ```text
/// w_eff = ((base * excitatory - base * inhibitory) * (1 + (modulatory - 0.5) * 0.4)).clamp(0.0, 1.0)
/// ```
///
/// # Example
/// ```rust
/// use context_graph_graph::marblestone::compute_effective_validated;
/// use context_graph_graph::{NeurotransmitterWeights, Domain};
///
/// let weights = NeurotransmitterWeights::for_domain(Domain::General);
/// let effective = compute_effective_validated(&weights, 1.0).unwrap();
/// // General: 0.276 ± 0.001
/// assert!((effective - 0.276).abs() < 0.001);
/// ```
pub fn compute_effective_validated(
    weights: &NeurotransmitterWeights,
    base_weight: f32,
) -> GraphResult<f32> {
    validate_or_error(weights)?;
    Ok(weights.compute_effective_weight(base_weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::marblestone::Domain;

    #[test]
    fn test_validate_or_error_valid_weights() {
        let valid = NeurotransmitterWeights::new(0.5, 0.3, 0.4);
        assert!(validate_or_error(&valid).is_ok());
    }

    #[test]
    fn test_validate_or_error_invalid_excitatory() {
        let invalid = NeurotransmitterWeights::new(1.5, 0.5, 0.5);
        let result = validate_or_error(&invalid);
        match result {
            Err(GraphError::InvalidNtWeights { field, value }) => {
                assert_eq!(field, "excitatory");
                assert!((value - 1.5).abs() < 0.001);
            }
            _ => panic!("Expected InvalidNtWeights error"),
        }
    }

    #[test]
    fn test_validate_or_error_invalid_inhibitory() {
        let invalid = NeurotransmitterWeights::new(0.5, -0.1, 0.5);
        let result = validate_or_error(&invalid);
        match result {
            Err(GraphError::InvalidNtWeights { field, value }) => {
                assert_eq!(field, "inhibitory");
                assert!((value - (-0.1)).abs() < 0.001);
            }
            _ => panic!("Expected InvalidNtWeights error"),
        }
    }

    #[test]
    fn test_validate_or_error_invalid_modulatory() {
        let invalid = NeurotransmitterWeights::new(0.5, 0.5, 1.5);
        let result = validate_or_error(&invalid);
        match result {
            Err(GraphError::InvalidNtWeights { field, value }) => {
                assert_eq!(field, "modulatory");
                assert!((value - 1.5).abs() < 0.001);
            }
            _ => panic!("Expected InvalidNtWeights error"),
        }
    }

    #[test]
    fn test_compute_effective_validated_success() {
        let weights = NeurotransmitterWeights::for_domain(Domain::General);
        let result = compute_effective_validated(&weights, 1.0);
        assert!(result.is_ok());
        let effective = result.unwrap();
        assert!((effective - 0.276).abs() < 0.001);
    }

    #[test]
    fn test_compute_effective_validated_fails_on_invalid() {
        let invalid = NeurotransmitterWeights::new(-0.1, 0.5, 0.5);
        let result = compute_effective_validated(&invalid, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_question_mark_operator_propagates() {
        fn inner_fn() -> GraphResult<f32> {
            let weights = NeurotransmitterWeights::new(1.5, 0.0, 0.0);
            validate_or_error(&weights)?; // Should return early
            Ok(42.0) // Should not reach here
        }

        let result = inner_fn();
        assert!(matches!(result, Err(GraphError::InvalidNtWeights { .. })));
    }
}
