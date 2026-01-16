//! Edge Case Tests

use super::helpers::*;
use context_graph_embeddings::{IndexEntry, StoredQuantizedFingerprint};
use uuid::Uuid;

/// Test get_embedding panics for invalid index.
#[test]
#[should_panic(expected = "STORAGE ERROR")]
fn test_get_embedding_invalid_index() {
    let fp = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(42),
        create_purpose_vector(42),
        create_johari_quadrants(42),
        create_content_hash(42),
    );

    let _ = fp.get_embedding(99); // Invalid index
}

/// Test very small vectors in IndexEntry.
#[test]
fn test_small_vector_values() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![1e-10, 1e-10, 1e-10]);
    let normalized = entry.normalized();

    // Should still produce valid normalized vector
    let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3 || norm == 0.0,
        "Normalized very small vector should have unit norm or be zero"
    );

    println!("[PASS] Small vector values handled correctly");
}

/// Test very large vector values in IndexEntry.
#[test]
fn test_large_vector_values() {
    let entry = IndexEntry::new(Uuid::new_v4(), 0, vec![1e10, 1e10, 1e10]);
    let sim = entry.cosine_similarity(&[1e10, 1e10, 1e10]);

    assert!(
        (sim - 1.0).abs() < 1e-3,
        "Large identical vectors should have similarity ~1.0, got {}",
        sim
    );

    println!("[PASS] Large vector values handled correctly");
}

/// Test fingerprint with all equal purpose values.
#[test]
fn test_uniform_purpose_vector() {
    let fp = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(42),
        [0.5f32; 13],
        create_johari_quadrants(42),
        create_content_hash(42),
    );

    // TASK-P0-001: alignment_score field removed per ARCH-03
    // Verify purpose_vector is correctly stored instead
    assert_eq!(fp.purpose_vector, [0.5f32; 13]);

    println!("[PASS] Uniform purpose vector handled correctly");
}

/// Test fingerprint with extreme johari quadrant (one dominant).
#[test]
fn test_extreme_johari_quadrants() {
    let fp = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(42),
        create_purpose_vector(42),
        [1.0, 0.0, 0.0, 0.0], // 100% Open
        create_content_hash(42),
    );

    assert_eq!(fp.dominant_quadrant, 0, "Open should be dominant");
    assert!(
        (fp.johari_confidence - 1.0).abs() < f32::EPSILON,
        "Confidence should be 1.0"
    );

    println!("[PASS] Extreme johari quadrants handled correctly");
}

/// Test fingerprint with all zero johari quadrants.
#[test]
fn test_zero_johari_quadrants() {
    let fp = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(42),
        create_purpose_vector(42),
        [0.0, 0.0, 0.0, 0.0],
        create_content_hash(42),
    );

    // Should default to quadrant 0 with zero confidence
    assert_eq!(
        fp.dominant_quadrant, 0,
        "Should default to Open with zero input"
    );
    assert!(
        (fp.johari_confidence - 0.0).abs() < f32::EPSILON,
        "Confidence should be 0.0"
    );

    println!("[PASS] Zero johari quadrants handled correctly");
}

/// Test multiple roundtrips don't accumulate errors.
#[test]
fn test_multiple_roundtrips_no_drift() {
    let original = StoredQuantizedFingerprint::new(
        Uuid::new_v4(),
        create_test_embeddings_with_deterministic_data(42),
        create_purpose_vector(42),
        create_johari_quadrants(42),
        create_content_hash(42),
    );

    // 10 roundtrips
    let mut current = original.clone();
    for _ in 0..10 {
        let json = serde_json::to_string(&current).expect("Serialize");
        current = serde_json::from_str(&json).expect("Deserialize");
    }

    // Verify no drift
    assert_eq!(current.id, original.id, "ID drifted after 10 roundtrips");
    assert_eq!(
        current.content_hash, original.content_hash,
        "Content hash drifted"
    );
    assert_eq!(
        current.purpose_vector, original.purpose_vector,
        "Purpose vector drifted"
    );

    for i in 0..13u8 {
        let orig_data = &original.get_embedding(i).data;
        let curr_data = &current.get_embedding(i).data;
        assert_eq!(
            orig_data, curr_data,
            "Embedder {} data drifted after 10 roundtrips",
            i
        );
    }

    println!("[PASS] 10 roundtrips with no data drift");
}
