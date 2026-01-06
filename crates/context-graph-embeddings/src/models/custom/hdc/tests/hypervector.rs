//! Random hypervector generation tests.

use super::*;

#[test]
fn test_random_hypervector_dimension() {
    let model = HdcModel::default_model();
    let hv = model.random_hypervector(42);
    assert_eq!(hv.len(), HDC_DIMENSION);
}

#[test]
fn test_random_hypervector_approximately_50_percent_ones() {
    let model = HdcModel::default_model();
    let hv = model.random_hypervector(123);
    let ones = hv.count_ones();
    // Should be roughly 50% +/- 5%
    let lower = (HDC_DIMENSION as f32 * 0.45) as usize;
    let upper = (HDC_DIMENSION as f32 * 0.55) as usize;
    assert!(
        ones >= lower && ones <= upper,
        "Expected ~50% ones, got {} ({}%)",
        ones,
        (ones as f32 / HDC_DIMENSION as f32) * 100.0
    );
}

#[test]
fn test_random_hypervector_deterministic() {
    let model = HdcModel::default_model();
    let hv1 = model.random_hypervector(999);
    let hv2 = model.random_hypervector(999);
    assert_eq!(hv1, hv2);
}

#[test]
fn test_random_hypervector_different_keys_differ() {
    let model = HdcModel::default_model();
    let hv1 = model.random_hypervector(1);
    let hv2 = model.random_hypervector(2);
    assert_ne!(hv1, hv2);
    // They should be approximately orthogonal (Hamming distance ~50%)
    let similarity = HdcModel::similarity(&hv1, &hv2);
    assert!(
        similarity > 0.45 && similarity < 0.55,
        "Expected ~0.5 similarity, got {}",
        similarity
    );
}

#[test]
fn test_random_hypervector_different_seeds_differ() {
    let model1 = HdcModel::new(3, 100).unwrap();
    let model2 = HdcModel::new(3, 200).unwrap();
    let hv1 = model1.random_hypervector(42);
    let hv2 = model2.random_hypervector(42);
    assert_ne!(hv1, hv2);
}
