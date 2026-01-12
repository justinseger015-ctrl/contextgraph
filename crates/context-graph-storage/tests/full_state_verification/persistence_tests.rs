//! Persistence Tests
//!
//! Tests for update/delete operations, database reopen, and ego_node persistence.

use context_graph_core::traits::TeleologicalMemoryStore;
use context_graph_storage::teleological::{
    deserialize_teleological_fingerprint, fingerprint_key, purpose_vector_key,
    RocksDbTeleologicalStore, CF_FINGERPRINTS, CF_PURPOSE_VECTORS,
};
use tempfile::TempDir;
use uuid::Uuid;

use crate::helpers::{create_test_store, generate_real_teleological_fingerprint, hex_string};

/// Test 7: Update and Delete Physical Verification
///
/// Tests that updates and deletes are physically reflected in RocksDB.
#[tokio::test]
async fn test_update_delete_physical_verification() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Update and Delete");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let store = create_test_store(&temp_dir);

    let id = Uuid::new_v4();
    let mut fingerprint = generate_real_teleological_fingerprint(id);

    // Initial store
    let original_alignments = fingerprint.purpose_vector.alignments;
    store
        .store(fingerprint.clone())
        .await
        .expect("Initial store failed");

    println!("[1] Initial store:");
    println!("    Alignments[0]: {:.6}", original_alignments[0]);

    // Verify initial state
    let key = fingerprint_key(&id);
    let raw1 = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Read failed")
        .expect("Not found");
    let retrieved1 = deserialize_teleological_fingerprint(&raw1);

    println!("[2] Physical verification of initial state:");
    println!(
        "    Alignments[0]: {:.6}",
        retrieved1.purpose_vector.alignments[0]
    );
    assert!((retrieved1.purpose_vector.alignments[0] - original_alignments[0]).abs() < 0.0001);

    // UPDATE: Change purpose vector
    fingerprint.purpose_vector.alignments[0] = 0.999;
    store
        .update(fingerprint.clone())
        .await
        .expect("Update failed");

    println!("[3] Update applied:");
    println!(
        "    New alignments[0]: {:.6}",
        fingerprint.purpose_vector.alignments[0]
    );

    // Physical verification after update
    let raw2 = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Read failed")
        .expect("Not found after update");
    let retrieved2 = deserialize_teleological_fingerprint(&raw2);

    println!("[4] Physical verification after update:");
    println!(
        "    Alignments[0]: {:.6}",
        retrieved2.purpose_vector.alignments[0]
    );

    assert!((retrieved2.purpose_vector.alignments[0] - 0.999).abs() < 0.001);

    // DELETE (hard delete)
    store.delete(id, false).await.expect("Delete failed");
    println!("[5] Delete executed");

    // Physical verification after delete
    let raw3 = store
        .get_raw_bytes(CF_FINGERPRINTS, &key)
        .expect("Read failed");

    println!("[6] Physical verification after delete:");
    println!("    Data exists: {}", raw3.is_some());

    assert!(raw3.is_none(), "Fingerprint still exists after delete!");

    // Verify purpose vector CF also deleted
    let purpose_key = purpose_vector_key(&id);
    let raw_purpose = store
        .get_raw_bytes(CF_PURPOSE_VECTORS, &purpose_key)
        .expect("Read failed");

    println!("[7] Purpose vector CF after delete:");
    println!("    Data exists: {}", raw_purpose.is_some());

    assert!(raw_purpose.is_none(), "Purpose vector still exists!");

    println!("\n[PASS] Update and delete physical verification successful");
    println!("================================================================================\n");
}

/// Test 9: Persistence Across DB Reopen
///
/// Verifies data survives database close and reopen.
#[tokio::test]
async fn test_persistence_across_reopen() {
    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: Persistence Across Reopen");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().to_path_buf();

    let id = Uuid::new_v4();
    let fingerprint = generate_real_teleological_fingerprint(id);
    let original_alignments = fingerprint.purpose_vector.alignments;

    // First session: store data
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to open store");
        // Note: EmbedderIndexRegistry is initialized in constructor

        store.store(fingerprint).await.expect("Store failed");
        println!("[1] First session: stored fingerprint {}", id);

        // Explicit drop to close DB
        drop(store);
        println!("[2] First session: database closed");
    }

    // Second session: reopen and verify
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to reopen store");
        // Note: EmbedderIndexRegistry is initialized in constructor

        println!("[3] Second session: database reopened");

        // Physical verification
        let key = fingerprint_key(&id);
        let raw = store
            .get_raw_bytes(CF_FINGERPRINTS, &key)
            .expect("Read failed");

        assert!(raw.is_some(), "Data lost after reopen!");
        let bytes = raw.unwrap();

        let retrieved = deserialize_teleological_fingerprint(&bytes);

        println!("[4] Physical verification after reopen:");
        println!("    ID match: {}", retrieved.id == id);
        println!(
            "    Alignments match: {}",
            retrieved.purpose_vector.alignments == original_alignments
        );

        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.purpose_vector.alignments, original_alignments);
    }

    println!("\n[PASS] Persistence verification successful");
    println!("================================================================================\n");
}

// =============================================================================
// TASK-GWT-P1-001: SELF_EGO_NODE PERSISTENCE FSV
// =============================================================================

/// Test 10: SELF_EGO_NODE Physical Persistence Verification
///
/// FULL STATE VERIFICATION that physically verifies ego_node data exists
/// in RocksDB after save_ego_node operations.
///
/// This test performs the 6 verification steps:
/// 1. Create temp RocksDB store
/// 2. Save SelfEgoNode with KNOWN synthetic data
/// 3. PHYSICALLY read raw bytes from RocksDB's ego_node CF using get_cf()
/// 4. Verify raw bytes exist and are not empty
/// 5. Deserialize and verify data matches input
/// 6. Close store, reopen, verify persistence
#[tokio::test]
async fn test_fsv_ego_node_physical_persistence() {
    use context_graph_core::gwt::ego_node::SelfEgoNode;
    use context_graph_storage::teleological::{deserialize_ego_node, ego_node_key, CF_EGO_NODE};

    println!("\n================================================================================");
    println!("FULL STATE VERIFICATION: SELF_EGO_NODE Physical Persistence (TASK-GWT-P1-001)");
    println!("================================================================================\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let path = temp_dir.path().to_path_buf();

    // Create KNOWN SYNTHETIC DATA per task requirements
    let synthetic_id = Uuid::nil(); // Known UUID
    let synthetic_purpose_vector: [f32; 13] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.11, 0.12, 0.13,
    ];
    let synthetic_coherence: f32 = 0.77;

    println!("[1] SYNTHETIC DATA DEFINITION");
    println!("    id: {} (Uuid::nil())", synthetic_id);
    println!(
        "    purpose_vector: [{:.2}, {:.2}, {:.2}, ..., {:.2}]",
        synthetic_purpose_vector[0],
        synthetic_purpose_vector[1],
        synthetic_purpose_vector[2],
        synthetic_purpose_vector[12]
    );
    println!("    coherence_with_actions: {:.2}", synthetic_coherence);

    // Create SelfEgoNode with known data
    let mut ego_node = SelfEgoNode::new();
    ego_node.id = synthetic_id;
    ego_node.purpose_vector = synthetic_purpose_vector;
    ego_node.coherence_with_actions = synthetic_coherence;

    // ===========================================================================
    // STEP 1: BEFORE STATE - Verify no ego_node exists
    // ===========================================================================
    let bytes_size: usize;
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to open store");

        println!("\n[2] BEFORE STATE - Verify empty");
        let key = ego_node_key();
        let before_raw = store
            .get_raw_bytes(CF_EGO_NODE, key)
            .expect("Failed to read raw bytes");

        println!("    Key (hex): {}", hex_string(key));
        println!("    Data exists: {}", before_raw.is_some());

        assert!(
            before_raw.is_none(),
            "FAIL: Ego node should NOT exist before save!"
        );
        println!("    [OK] No ego_node in database - BEFORE state verified");
    }

    // ===========================================================================
    // STEP 2: EXECUTE - Save ego_node
    // ===========================================================================
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to reopen store");

        println!("\n[3] EXECUTE - Save ego_node");
        store
            .save_ego_node(&ego_node)
            .await
            .expect("Failed to save ego node");
        store.flush().await.expect("Failed to flush");

        println!("    save_ego_node() called: SUCCESS");
        println!("    flush() called: SUCCESS");
    }

    // ===========================================================================
    // STEP 3: AFTER STATE - Physical read via get_cf()
    // ===========================================================================
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to reopen store");

        println!("\n[4] AFTER STATE - Physical Verification via get_raw_bytes()");
        let key = ego_node_key();
        let after_raw = store
            .get_raw_bytes(CF_EGO_NODE, key)
            .expect("Failed to read raw bytes");

        assert!(
            after_raw.is_some(),
            "FAIL: Ego node NOT found in RocksDB after save!"
        );
        let raw_bytes = after_raw.unwrap();
        bytes_size = raw_bytes.len();

        // ===========================================================================
        // EVIDENCE: Physical byte-level verification
        // ===========================================================================
        println!("\n[5] PHYSICAL EVIDENCE");
        println!("    Key: \"ego_node\" (8 bytes)");
        println!("    Key (hex): {}", hex_string(key));
        println!("    Value size: {} bytes", raw_bytes.len());
        println!(
            "    First 32 bytes (hex): {}",
            hex_string(&raw_bytes[..32.min(raw_bytes.len())])
        );
        println!("    Version byte: 0x{:02x} (expected 0x01)", raw_bytes[0]);

        // Verify version byte
        assert_eq!(
            raw_bytes[0], 1,
            "FAIL: Version byte should be 1, got {}",
            raw_bytes[0]
        );
        println!("    [OK] Version byte is 0x01 (EGO_NODE_VERSION)");

        // Verify minimum size (50 bytes per serialization.rs)
        assert!(
            raw_bytes.len() >= 50,
            "FAIL: Raw bytes too small: {} (expected >= 50)",
            raw_bytes.len()
        );
        println!(
            "    [OK] Value size {} >= 50 bytes (MIN_EGO_NODE_SIZE)",
            raw_bytes.len()
        );

        // ===========================================================================
        // STEP 4: Deserialize and verify data matches
        // ===========================================================================
        println!("\n[6] DESERIALIZATION VERIFICATION");
        let deserialized = deserialize_ego_node(&raw_bytes);

        println!("    Deserialized id: {}", deserialized.id);
        println!(
            "    Deserialized purpose_vector[0..3]: [{:.2}, {:.2}, {:.2}]",
            deserialized.purpose_vector[0],
            deserialized.purpose_vector[1],
            deserialized.purpose_vector[2]
        );
        println!(
            "    Deserialized coherence: {:.2}",
            deserialized.coherence_with_actions
        );

        // Verify exact match
        assert_eq!(
            deserialized.id, synthetic_id,
            "FAIL: ID mismatch! Expected {}, got {}",
            synthetic_id, deserialized.id
        );
        println!("    [OK] ID matches synthetic data");

        assert_eq!(
            deserialized.purpose_vector, synthetic_purpose_vector,
            "FAIL: Purpose vector mismatch!"
        );
        println!("    [OK] Purpose vector matches synthetic data");

        assert!(
            (deserialized.coherence_with_actions - synthetic_coherence).abs() < 0.0001,
            "FAIL: Coherence mismatch! Expected {}, got {}",
            synthetic_coherence,
            deserialized.coherence_with_actions
        );
        println!("    [OK] Coherence matches synthetic data");
    }

    // ===========================================================================
    // STEP 5: Close and reopen - Verify persistence survives
    // ===========================================================================
    println!("\n[7] PERSISTENCE ACROSS REOPEN");

    // Reopen same path and verify
    {
        let store = RocksDbTeleologicalStore::open(&path).expect("Failed to reopen store");

        let loaded = store
            .load_ego_node()
            .await
            .expect("Failed to load ego node")
            .expect("Ego node should exist after reopen");

        assert_eq!(loaded.id, synthetic_id, "FAIL: ID lost after reopen!");
        assert_eq!(
            loaded.purpose_vector, synthetic_purpose_vector,
            "FAIL: Purpose vector lost after reopen!"
        );
        assert!(
            (loaded.coherence_with_actions - synthetic_coherence).abs() < 0.0001,
            "FAIL: Coherence lost after reopen!"
        );

        println!("    Session 2: Reopened and verified all data persisted");
        println!("    [OK] ID: {}", loaded.id);
        println!("    [OK] Purpose vector: matches");
        println!("    [OK] Coherence: {:.2}", loaded.coherence_with_actions);
    }

    // ===========================================================================
    // SUMMARY
    // ===========================================================================
    println!("\n================================================================================");
    println!("[PASS] SELF_EGO_NODE Physical Persistence Verification COMPLETE");
    println!("================================================================================");
    println!("\nEVIDENCE SUMMARY:");
    println!("  - BEFORE: get_raw_bytes(CF_EGO_NODE) returned None");
    println!("  - EXECUTE: save_ego_node() succeeded");
    println!(
        "  - AFTER: get_raw_bytes(CF_EGO_NODE) returned Some({} bytes)",
        bytes_size
    );
    println!("  - VERSION: First byte = 0x01 (EGO_NODE_VERSION)");
    println!("  - DESERIALIZE: All fields match synthetic input");
    println!("  - REOPEN: Data persisted across database close/reopen cycle");
    println!("\n");
}
