//! Text encoding and projection tests.

use super::*;
use bitvec::prelude::*;

// ========================================================================
// TEXT ENCODING TESTS
// ========================================================================

#[test]
fn test_encode_text_produces_correct_dimension() {
    let model = HdcModel::default_model();
    let hv = model.encode_text("hello world");
    assert_eq!(hv.len(), HDC_DIMENSION);
}

#[test]
fn test_encode_text_deterministic() {
    let model = HdcModel::default_model();
    let hv1 = model.encode_text("The quick brown fox");
    let hv2 = model.encode_text("The quick brown fox");
    assert_eq!(hv1, hv2);
}

#[test]
fn test_encode_text_different_strings_differ() {
    let model = HdcModel::default_model();
    let hv1 = model.encode_text("hello");
    let hv2 = model.encode_text("world");
    assert_ne!(hv1, hv2);
}

#[test]
fn test_encode_text_similar_strings_similar() {
    let model = HdcModel::default_model();
    let hv1 = model.encode_text("hello world");
    let hv2 = model.encode_text("hello world!");
    let sim = HdcModel::similarity(&hv1, &hv2);
    // Similar strings should have higher similarity than random
    assert!(
        sim > 0.6,
        "Similar strings should have high similarity: {}",
        sim
    );
}

#[test]
fn test_encode_text_empty_returns_zero() {
    let model = HdcModel::default_model();
    let hv = model.encode_text("");
    assert_eq!(hv.count_ones(), 0);
}

// ========================================================================
// PROJECTION TESTS
// ========================================================================

#[test]
fn test_project_to_float_dimension() {
    let model = HdcModel::default_model();
    let hv = model.random_hypervector(42);
    let projected = model.project_to_float(&hv);
    assert_eq!(projected.len(), HDC_PROJECTED_DIMENSION);
}

#[test]
fn test_project_to_float_normalized() {
    let model = HdcModel::default_model();
    let hv = model.random_hypervector(42);
    let projected = model.project_to_float(&hv);

    let norm: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.001,
        "Projected vector should be L2 normalized, got norm {}",
        norm
    );
}

#[test]
fn test_project_to_float_values_in_range() {
    let model = HdcModel::default_model();
    let hv = model.random_hypervector(42);
    let projected = model.project_to_float(&hv);

    for (i, &val) in projected.iter().enumerate() {
        assert!(
            (-1.5..=1.5).contains(&val),
            "Value at {} = {} out of expected range",
            i,
            val
        );
        assert!(val.is_finite(), "Value at {} is not finite", i);
    }
}

#[test]
fn test_project_to_float_all_zeros_handled() {
    let model = HdcModel::default_model();
    let zero_hv = bitvec![u64, Lsb0; 0; HDC_DIMENSION];
    let projected = model.project_to_float(&zero_hv);
    assert_eq!(projected.len(), HDC_PROJECTED_DIMENSION);
    // All values should be finite (handles division by zero)
    for val in &projected {
        assert!(val.is_finite());
    }
}
