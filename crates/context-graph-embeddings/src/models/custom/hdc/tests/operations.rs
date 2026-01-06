//! Bind, bundle, permute, and similarity operation tests.

use super::*;

// ========================================================================
// BIND (XOR) TESTS
// ========================================================================

#[test]
fn test_bind_self_inverse() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(1);
    let b = model.random_hypervector(2);
    let bound = HdcModel::bind(&a, &b);
    let unbound = HdcModel::bind(&bound, &b);
    assert_eq!(a, unbound, "A ^ B ^ B should equal A");
}

#[test]
fn test_bind_commutative() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(10);
    let b = model.random_hypervector(20);
    let ab = HdcModel::bind(&a, &b);
    let ba = HdcModel::bind(&b, &a);
    assert_eq!(ab, ba, "XOR should be commutative");
}

#[test]
fn test_bind_produces_orthogonal() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(100);
    let b = model.random_hypervector(200);
    let bound = HdcModel::bind(&a, &b);

    // Bound should be approximately orthogonal to both inputs
    let sim_a = HdcModel::similarity(&bound, &a);
    let sim_b = HdcModel::similarity(&bound, &b);
    assert!(
        sim_a > 0.45 && sim_a < 0.55,
        "Bound should be ~orthogonal to A, got {}",
        sim_a
    );
    assert!(
        sim_b > 0.45 && sim_b < 0.55,
        "Bound should be ~orthogonal to B, got {}",
        sim_b
    );
}

#[test]
fn test_bind_with_self_is_zero() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(50);
    let result = HdcModel::bind(&a, &a);
    assert_eq!(result.count_ones(), 0, "A ^ A should be all zeros");
}

// ========================================================================
// BUNDLE (MAJORITY) TESTS
// ========================================================================

#[test]
fn test_bundle_empty_returns_zero_vector() {
    let result = HdcModel::bundle(&[]);
    assert_eq!(result.len(), HDC_DIMENSION);
    assert_eq!(result.count_ones(), 0);
}

#[test]
fn test_bundle_single_vector_returns_copy() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(42);
    let result = HdcModel::bundle(std::slice::from_ref(&v));
    assert_eq!(result, v);
}

#[test]
fn test_bundle_two_vectors_majority() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(1);
    let b = model.random_hypervector(2);
    let bundled = HdcModel::bundle(&[a.clone(), b.clone()]);

    // With two vectors, ties are broken by first vector
    // So bundled should be similar to both, leaning toward a
    let sim_a = HdcModel::similarity(&bundled, &a);
    let sim_b = HdcModel::similarity(&bundled, &b);
    // Both should have some similarity (not orthogonal)
    assert!(sim_a > 0.45, "Should have some similarity to A: {}", sim_a);
    assert!(sim_b > 0.45, "Should have some similarity to B: {}", sim_b);
}

#[test]
fn test_bundle_three_same_vectors_returns_same() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(77);
    let bundled = HdcModel::bundle(&[v.clone(), v.clone(), v.clone()]);
    assert_eq!(
        bundled, v,
        "Bundle of 3 identical vectors should be identical"
    );
}

#[test]
fn test_bundle_preserves_similarity() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(1);
    let b = model.random_hypervector(2);
    let c = model.random_hypervector(3);

    // Bundle a, b, c
    let bundled = HdcModel::bundle(&[a.clone(), b.clone(), c.clone()]);

    // Bundled should be more similar to each input than random vectors are to each other
    let sim_a = HdcModel::similarity(&bundled, &a);
    let sim_b = HdcModel::similarity(&bundled, &b);
    let sim_c = HdcModel::similarity(&bundled, &c);

    // Each should have higher than orthogonal similarity
    assert!(sim_a > 0.55, "Similarity to A: {}", sim_a);
    assert!(sim_b > 0.55, "Similarity to B: {}", sim_b);
    assert!(sim_c > 0.55, "Similarity to C: {}", sim_c);
}

// ========================================================================
// PERMUTE TESTS
// ========================================================================

#[test]
fn test_permute_zero_shift_identity() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(42);
    let result = HdcModel::permute(&v, 0);
    assert_eq!(result, v);
}

#[test]
fn test_permute_full_rotation_identity() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(42);
    let result = HdcModel::permute(&v, HDC_DIMENSION);
    assert_eq!(result, v);
}

#[test]
fn test_permute_preserves_popcount() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(42);
    let original_ones = v.count_ones();

    for shift in [1, 10, 100, 1000, 5000] {
        let result = HdcModel::permute(&v, shift);
        assert_eq!(
            result.count_ones(),
            original_ones,
            "Permute by {} should preserve popcount",
            shift
        );
    }
}

#[test]
fn test_permute_different_shifts_differ() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(42);
    let p1 = HdcModel::permute(&v, 1);
    let p2 = HdcModel::permute(&v, 2);
    assert_ne!(p1, p2);
    assert_ne!(p1, v);
}

// ========================================================================
// HAMMING DISTANCE & SIMILARITY TESTS
// ========================================================================

#[test]
fn test_hamming_distance_identical_is_zero() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(42);
    assert_eq!(HdcModel::hamming_distance(&v, &v), 0);
}

#[test]
fn test_similarity_identical_is_one() {
    let model = HdcModel::default_model();
    let v = model.random_hypervector(42);
    assert_eq!(HdcModel::similarity(&v, &v), 1.0);
}

#[test]
fn test_similarity_orthogonal_is_about_half() {
    let model = HdcModel::default_model();
    let a = model.random_hypervector(1);
    let b = model.random_hypervector(2);
    let sim = HdcModel::similarity(&a, &b);
    assert!(
        sim > 0.45 && sim < 0.55,
        "Random vectors should have ~0.5 similarity, got {}",
        sim
    );
}
