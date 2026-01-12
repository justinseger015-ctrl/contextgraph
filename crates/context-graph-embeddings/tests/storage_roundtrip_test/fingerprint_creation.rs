//! Fingerprint Creation Tests

use super::helpers::*;
use context_graph_embeddings::{
    ModelId, QuantizationMethod, StoredQuantizedFingerprint, MAX_QUANTIZED_SIZE_BYTES,
    STORAGE_VERSION,
};
use uuid::Uuid;

/// Test creating fingerprint with all 13 embeddings succeeds.
#[test]
fn test_create_fingerprint_with_all_13_embeddings() {
    let id = Uuid::new_v4();
    let embeddings = create_test_embeddings_with_deterministic_data(42);
    let purpose_vector = create_purpose_vector(42);
    let johari_quadrants = create_johari_quadrants(42);
    let content_hash = create_content_hash(42);

    let fp = StoredQuantizedFingerprint::new(
        id,
        embeddings,
        purpose_vector,
        johari_quadrants,
        content_hash,
    );

    // Verify all fields
    assert_eq!(fp.id, id, "ID must match");
    assert_eq!(
        fp.version, STORAGE_VERSION,
        "Version must match STORAGE_VERSION"
    );
    assert_eq!(fp.embeddings.len(), 13, "Must have exactly 13 embeddings");
    assert_eq!(
        fp.purpose_vector.len(),
        13,
        "Purpose vector must have 13 dimensions"
    );
    assert_eq!(
        fp.johari_quadrants.len(),
        4,
        "Johari quadrants must have 4 values"
    );
    assert_eq!(fp.content_hash, content_hash, "Content hash must match");
    assert!(!fp.deleted, "New fingerprint should not be deleted");
    assert_eq!(
        fp.access_count, 0,
        "New fingerprint should have zero access count"
    );

    println!("[PASS] Created fingerprint with all 13 embeddings");
}

/// Test that missing any single embedding panics.
#[test]
#[should_panic(expected = "CONSTRUCTION ERROR")]
fn test_panic_on_missing_embedder_0() {
    let mut embeddings = create_test_embeddings_with_deterministic_data(42);
    embeddings.remove(&0);

    let _ = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        embeddings,
        create_purpose_vector(42),
        create_johari_quadrants(42),
        create_content_hash(42),
    );
}

#[test]
#[should_panic(expected = "CONSTRUCTION ERROR")]
fn test_panic_on_missing_embedder_6() {
    let mut embeddings = create_test_embeddings_with_deterministic_data(42);
    embeddings.remove(&6);

    let _ = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        embeddings,
        create_purpose_vector(42),
        create_johari_quadrants(42),
        create_content_hash(42),
    );
}

#[test]
#[should_panic(expected = "CONSTRUCTION ERROR")]
fn test_panic_on_missing_embedder_12() {
    let mut embeddings = create_test_embeddings_with_deterministic_data(42);
    embeddings.remove(&12);

    let _ = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        embeddings,
        create_purpose_vector(42),
        create_johari_quadrants(42),
        create_content_hash(42),
    );
}

/// Test that each embedding has correct quantization method.
#[test]
fn test_embeddings_have_correct_quantization_methods() {
    let embeddings = create_test_embeddings_with_deterministic_data(42);

    let fp = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        embeddings,
        create_purpose_vector(42),
        create_johari_quadrants(42),
        create_content_hash(42),
    );

    // Verify each embedder uses Constitution-correct method
    for i in 0..13u8 {
        let model_id = ModelId::try_from(i).expect("Valid model index");
        let expected_method = QuantizationMethod::for_model_id(model_id);
        let actual_method = fp.get_embedding(i).method;
        assert_eq!(
            actual_method, expected_method,
            "Embedder {} should use {:?}, got {:?}",
            i, expected_method, actual_method
        );
    }

    assert!(
        fp.validate_quantization_methods(),
        "All quantization methods should be valid"
    );
    println!("[PASS] All 13 embeddings have correct quantization methods");
}

/// Test theta_to_north_star is computed correctly.
#[test]
fn test_theta_to_north_star_computation() {
    let pv = [0.5f32; 13]; // Uniform purpose vector

    let fp = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(42),
        pv,
        create_johari_quadrants(42),
        create_content_hash(42),
    );

    // theta = mean of purpose vector = 0.5
    assert!(
        (fp.theta_to_north_star - 0.5).abs() < f32::EPSILON,
        "theta_to_north_star should be mean of purpose vector (0.5), got {}",
        fp.theta_to_north_star
    );

    println!("[PASS] theta_to_north_star computed correctly");
}

/// Test dominant quadrant computation.
#[test]
fn test_dominant_quadrant_computation() {
    // Open dominant
    let fp1 = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(1),
        create_purpose_vector(1),
        [0.6, 0.2, 0.1, 0.1],
        create_content_hash(1),
    );
    assert_eq!(fp1.dominant_quadrant, 0, "Open should be dominant");
    assert!(
        (fp1.johari_confidence - 0.6).abs() < f32::EPSILON,
        "Confidence should be 0.6"
    );

    // Hidden dominant
    let fp2 = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(2),
        create_purpose_vector(2),
        [0.1, 0.7, 0.1, 0.1],
        create_content_hash(2),
    );
    assert_eq!(fp2.dominant_quadrant, 1, "Hidden should be dominant");

    // Unknown dominant
    let fp3 = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(3),
        create_purpose_vector(3),
        [0.1, 0.1, 0.2, 0.6],
        create_content_hash(3),
    );
    assert_eq!(fp3.dominant_quadrant, 3, "Unknown should be dominant");

    println!("[PASS] Dominant quadrant computed correctly for all cases");
}

/// Test estimated size is within Constitution bounds.
#[test]
fn test_estimated_size_within_bounds() {
    let fp = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(42),
        create_purpose_vector(42),
        create_johari_quadrants(42),
        create_content_hash(42),
    );

    let size = fp.estimated_size_bytes();

    // Must be less than MAX_QUANTIZED_SIZE_BYTES (25KB)
    assert!(
        size < MAX_QUANTIZED_SIZE_BYTES,
        "Estimated size {} exceeds maximum {} bytes",
        size,
        MAX_QUANTIZED_SIZE_BYTES
    );

    // Should be reasonable (> 1KB for all that data)
    assert!(
        size > 1000,
        "Estimated size {} seems too small for 13 embeddings",
        size
    );

    println!("[PASS] Estimated size {} bytes is within bounds", size);
}
