//! Integration tests for RocksDB storage column families.
//!
//! These tests use REAL RocksDB instances - NO MOCKS.
//! Each test creates a temporary database to verify actual functionality.
//!
//! # Constitution Reference
//!
//! - testing.types.integration: tests/integration/ - MCP, graph, session
//! - AP-007: Stub data in prod → use tests/fixtures/

use context_graph_graph::error::GraphError;
use context_graph_graph::storage::{
    get_column_family_descriptors, get_db_options, StorageConfig, ALL_COLUMN_FAMILIES,
    CF_ADJACENCY, CF_CONES, CF_FAISS_IDS, CF_HYPERBOLIC, CF_METADATA, CF_NODES,
};

// ========== Constants Tests ==========

#[test]
fn test_cf_names() {
    assert_eq!(CF_ADJACENCY, "adjacency");
    assert_eq!(CF_HYPERBOLIC, "hyperbolic");
    assert_eq!(CF_CONES, "entailment_cones");
    assert_eq!(CF_FAISS_IDS, "faiss_ids");
    assert_eq!(CF_NODES, "nodes");
    assert_eq!(CF_METADATA, "metadata");
}

#[test]
fn test_all_column_families_count() {
    assert_eq!(ALL_COLUMN_FAMILIES.len(), 6);
}

#[test]
fn test_all_column_families_contains_all() {
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_ADJACENCY));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_HYPERBOLIC));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_CONES));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_FAISS_IDS));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_NODES));
    assert!(ALL_COLUMN_FAMILIES.contains(&CF_METADATA));
}

// ========== StorageConfig Tests ==========

#[test]
fn test_storage_config_default() {
    let config = StorageConfig::default();
    assert_eq!(config.block_cache_size, 512 * 1024 * 1024);
    assert!(config.enable_compression);
    assert_eq!(config.bloom_filter_bits, 10);
    assert_eq!(config.write_buffer_size, 64 * 1024 * 1024);
    assert_eq!(config.max_write_buffers, 3);
    assert_eq!(config.target_file_size_base, 64 * 1024 * 1024);
}

#[test]
fn test_storage_config_read_optimized() {
    let config = StorageConfig::read_optimized();
    assert_eq!(config.block_cache_size, 1024 * 1024 * 1024); // 1GB
    assert_eq!(config.bloom_filter_bits, 14);
}

#[test]
fn test_storage_config_write_optimized() {
    let config = StorageConfig::write_optimized();
    assert_eq!(config.write_buffer_size, 128 * 1024 * 1024); // 128MB
    assert_eq!(config.max_write_buffers, 5);
}

#[test]
fn test_storage_config_validate_success() {
    let config = StorageConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_storage_config_validate_block_cache_too_small() {
    let config = StorageConfig {
        block_cache_size: 1024, // Only 1KB
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("block_cache_size"));
}

#[test]
fn test_storage_config_validate_bloom_filter_invalid() {
    let config = StorageConfig {
        bloom_filter_bits: 0, // Invalid
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("bloom_filter_bits"));
}

#[test]
fn test_storage_config_validate_write_buffer_too_small() {
    let config = StorageConfig {
        write_buffer_size: 512, // Only 512 bytes
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, GraphError::InvalidConfig(_)));
    assert!(err.to_string().contains("write_buffer_size"));
}

// ========== Column Family Descriptor Tests ==========

#[test]
fn test_get_column_family_descriptors_count() {
    let config = StorageConfig::default();
    let descriptors = get_column_family_descriptors(&config).unwrap();
    assert_eq!(descriptors.len(), 6);
}

#[test]
fn test_get_column_family_descriptors_invalid_config() {
    let config = StorageConfig {
        block_cache_size: 0,
        ..Default::default()
    };
    let result = get_column_family_descriptors(&config);
    assert!(result.is_err());
}

// ========== REAL RocksDB Integration Tests ==========

#[test]
fn test_real_rocksdb_open_with_column_families() {
    // REAL RocksDB - no mocks
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_cf.db");

    println!("BEFORE: Opening RocksDB at {:?}", db_path);

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    // Open REAL database
    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors)
        .expect("Failed to open RocksDB with column families");

    println!("AFTER: RocksDB opened successfully");

    // Verify all CFs exist
    for cf_name in ALL_COLUMN_FAMILIES {
        let cf_handle = db.cf_handle(cf_name);
        assert!(cf_handle.is_some(), "Column family {} must exist", cf_name);
        println!("VERIFIED: Column family '{}' exists", cf_name);
    }
}

#[test]
fn test_real_rocksdb_write_and_read_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_rw_metadata.db");

    println!("BEFORE: Opening RocksDB for write/read test at {:?}", db_path);

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write to metadata CF
    let metadata_cf = db.cf_handle(CF_METADATA).unwrap();
    db.put_cf(metadata_cf, b"schema_version", b"1").unwrap();

    println!("AFTER WRITE: Wrote schema_version=1 to metadata CF");

    // Read back
    let value = db.get_cf(metadata_cf, b"schema_version").unwrap();
    assert_eq!(value, Some(b"1".to_vec()));

    println!("AFTER READ: Verified schema_version=1");
}

#[test]
fn test_real_rocksdb_write_to_all_cfs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_all_cfs.db");

    println!("BEFORE: Opening RocksDB for all-CF write test at {:?}", db_path);

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write to each CF and verify
    for cf_name in ALL_COLUMN_FAMILIES {
        let cf = db.cf_handle(cf_name).unwrap();
        let key = format!("test_key_{}", cf_name);
        let value = format!("test_value_{}", cf_name);

        db.put_cf(cf, key.as_bytes(), value.as_bytes()).unwrap();

        let result = db.get_cf(cf, key.as_bytes()).unwrap();
        assert_eq!(result, Some(value.as_bytes().to_vec()));

        println!("VERIFIED: CF '{}' write/read successful", cf_name);
    }
}

#[test]
fn test_real_rocksdb_write_hyperbolic_coordinates() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_hyperbolic.db");

    println!("BEFORE: Testing hyperbolic coordinate storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Create 64D hyperbolic coordinates (256 bytes as per spec)
    let node_id: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let coordinates: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let coords_bytes: Vec<u8> = coordinates
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    assert_eq!(coords_bytes.len(), 256, "Hyperbolic coords must be 256 bytes");

    let hyperbolic_cf = db.cf_handle(CF_HYPERBOLIC).unwrap();
    db.put_cf(hyperbolic_cf, &node_id, &coords_bytes).unwrap();

    println!("AFTER WRITE: Stored 64D coordinates (256 bytes)");

    // Read back and verify
    let result = db.get_cf(hyperbolic_cf, &node_id).unwrap().unwrap();
    assert_eq!(result.len(), 256);

    // Deserialize and verify first value
    let first_f32 = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
    assert!((first_f32 - 0.0).abs() < 0.0001);

    println!("AFTER READ: Verified 256-byte hyperbolic coordinates");
}

#[test]
fn test_real_rocksdb_write_entailment_cone() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_cones.db");

    println!("BEFORE: Testing entailment cone storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Create entailment cone: 268 bytes (256 coords + 4 aperture + 4 factor + 4 depth)
    let node_id: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let mut cone_data: Vec<u8> = Vec::with_capacity(268);

    // 256 bytes for coordinates (64 f32)
    for i in 0..64 {
        cone_data.extend_from_slice(&(i as f32 * 0.01f32).to_le_bytes());
    }
    // 4 bytes for aperture
    cone_data.extend_from_slice(&0.5f32.to_le_bytes());
    // 4 bytes for factor
    cone_data.extend_from_slice(&1.0f32.to_le_bytes());
    // 4 bytes for depth
    cone_data.extend_from_slice(&3u32.to_le_bytes());

    assert_eq!(cone_data.len(), 268, "Cone data must be 268 bytes");

    let cones_cf = db.cf_handle(CF_CONES).unwrap();
    db.put_cf(cones_cf, &node_id, &cone_data).unwrap();

    println!("AFTER WRITE: Stored 268-byte entailment cone");

    // Read back and verify
    let result = db.get_cf(cones_cf, &node_id).unwrap().unwrap();
    assert_eq!(result.len(), 268);

    println!("AFTER READ: Verified 268-byte entailment cone");
}

#[test]
fn test_real_rocksdb_write_faiss_id() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_faiss_ids.db");

    println!("BEFORE: Testing FAISS ID mapping storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Store FAISS ID mapping (i64 = 8 bytes)
    let node_id: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let faiss_id: i64 = 42_000_000;
    let faiss_id_bytes = faiss_id.to_le_bytes();

    assert_eq!(faiss_id_bytes.len(), 8, "FAISS ID must be 8 bytes");

    let faiss_cf = db.cf_handle(CF_FAISS_IDS).unwrap();
    db.put_cf(faiss_cf, &node_id, &faiss_id_bytes).unwrap();

    println!("AFTER WRITE: Stored FAISS ID {}", faiss_id);

    // Read back and verify
    let result = db.get_cf(faiss_cf, &node_id).unwrap().unwrap();
    let read_id = i64::from_le_bytes([
        result[0], result[1], result[2], result[3],
        result[4], result[5], result[6], result[7],
    ]);
    assert_eq!(read_id, faiss_id);

    println!("AFTER READ: Verified FAISS ID {}", read_id);
}

#[test]
fn test_real_rocksdb_reopen_preserves_data() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_reopen.db");

    println!("BEFORE: Testing data persistence across reopen");

    let db_opts = get_db_options();
    let config = StorageConfig::default();

    // First open: write data
    {
        let cf_descriptors = get_column_family_descriptors(&config).unwrap();
        let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

        let nodes_cf = db.cf_handle(CF_NODES).unwrap();
        db.put_cf(nodes_cf, b"node_id_1", b"node_data_persistent").unwrap();

        println!("AFTER FIRST OPEN: Wrote node data");
    }

    // Second open: verify data persisted
    {
        let cf_descriptors = get_column_family_descriptors(&config).unwrap();
        let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

        let nodes_cf = db.cf_handle(CF_NODES).unwrap();
        let value = db.get_cf(nodes_cf, b"node_id_1").unwrap();
        assert_eq!(value, Some(b"node_data_persistent".to_vec()));

        println!("AFTER SECOND OPEN: Verified data persistence");
    }
}

#[test]
fn test_real_rocksdb_adjacency_prefix_scan() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_prefix_scan.db");

    println!("BEFORE: Testing adjacency prefix scan");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write multiple edges for the same source node
    let source_node: [u8; 16] = [1; 16];
    let adjacency_cf = db.cf_handle(CF_ADJACENCY).unwrap();

    // Store 3 edges from the same source
    for i in 0..3u8 {
        let mut key = source_node.to_vec();
        key.push(i); // Append edge index
        let value = format!("edge_to_target_{}", i);
        db.put_cf(adjacency_cf, &key, value.as_bytes()).unwrap();
    }

    println!("AFTER WRITE: Stored 3 edges from same source");

    // Use iterator to prefix scan
    let mut count = 0;
    let iter = db.prefix_iterator_cf(adjacency_cf, &source_node);
    for item in iter {
        let (key, _value) = item.unwrap();
        if key.starts_with(&source_node) {
            count += 1;
        } else {
            break;
        }
    }

    assert_eq!(count, 3, "Should find 3 edges with same prefix");
    println!("AFTER SCAN: Found {} edges with prefix scan", count);
}

#[test]
fn test_db_options_parallelism() {
    println!("BEFORE: Testing DB options with parallelism");

    let opts = get_db_options();

    // Verify options are valid by using them
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_opts.db");

    // Should succeed with our options
    let _db = rocksdb::DB::open(&opts, &db_path).unwrap();

    println!("AFTER: DB opened with parallelism options");
}

#[test]
fn test_storage_config_with_compression_disabled() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_no_compression.db");

    println!("BEFORE: Testing storage with compression disabled");

    let config = StorageConfig {
        enable_compression: false,
        ..Default::default()
    };

    let db_opts = get_db_options();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // Write and read to verify it works
    let nodes_cf = db.cf_handle(CF_NODES).unwrap();
    db.put_cf(nodes_cf, b"test_key", b"test_value").unwrap();

    let value = db.get_cf(nodes_cf, b"test_key").unwrap();
    assert_eq!(value, Some(b"test_value".to_vec()));

    println!("AFTER: Verified storage works without compression");
}

#[test]
fn test_storage_multiple_writes_same_key() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_overwrite.db");

    println!("BEFORE: Testing overwrite behavior");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    let metadata_cf = db.cf_handle(CF_METADATA).unwrap();

    // Write initial value
    db.put_cf(metadata_cf, b"version", b"1").unwrap();
    let v1 = db.get_cf(metadata_cf, b"version").unwrap();
    assert_eq!(v1, Some(b"1".to_vec()));

    // Overwrite
    db.put_cf(metadata_cf, b"version", b"2").unwrap();
    let v2 = db.get_cf(metadata_cf, b"version").unwrap();
    assert_eq!(v2, Some(b"2".to_vec()));

    // Overwrite again
    db.put_cf(metadata_cf, b"version", b"3").unwrap();
    let v3 = db.get_cf(metadata_cf, b"version").unwrap();
    assert_eq!(v3, Some(b"3".to_vec()));

    println!("AFTER: Verified overwrite behavior (1 -> 2 -> 3)");
}

#[test]
fn test_storage_delete_key() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_delete.db");

    println!("BEFORE: Testing delete behavior");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    let nodes_cf = db.cf_handle(CF_NODES).unwrap();

    // Write
    db.put_cf(nodes_cf, b"to_delete", b"value").unwrap();
    let exists = db.get_cf(nodes_cf, b"to_delete").unwrap();
    assert!(exists.is_some());

    // Delete
    db.delete_cf(nodes_cf, b"to_delete").unwrap();
    let deleted = db.get_cf(nodes_cf, b"to_delete").unwrap();
    assert!(deleted.is_none());

    println!("AFTER: Verified delete behavior");
}

#[test]
fn test_storage_nonexistent_key_returns_none() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_none.db");

    println!("BEFORE: Testing nonexistent key behavior");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    for cf_name in ALL_COLUMN_FAMILIES {
        let cf = db.cf_handle(cf_name).unwrap();
        let result = db.get_cf(cf, b"nonexistent_key_12345").unwrap();
        assert!(result.is_none(), "Nonexistent key should return None in {}", cf_name);
    }

    println!("AFTER: Verified all CFs return None for nonexistent keys");
}

// ========== Edge Case Tests ==========

#[test]
fn test_storage_empty_value() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_empty_value.db");

    println!("BEFORE: Testing empty value storage");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    let metadata_cf = db.cf_handle(CF_METADATA).unwrap();
    db.put_cf(metadata_cf, b"empty_key", b"").unwrap();

    let result = db.get_cf(metadata_cf, b"empty_key").unwrap();
    assert_eq!(result, Some(vec![]));

    println!("AFTER: Verified empty value storage");
}

#[test]
fn test_storage_large_value() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test_large_value.db");

    println!("BEFORE: Testing large value storage (1MB)");

    let db_opts = get_db_options();
    let config = StorageConfig::default();
    let cf_descriptors = get_column_family_descriptors(&config).unwrap();

    let db = rocksdb::DB::open_cf_descriptors(&db_opts, &db_path, cf_descriptors).unwrap();

    // 1MB value
    let large_value: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();

    let nodes_cf = db.cf_handle(CF_NODES).unwrap();
    db.put_cf(nodes_cf, b"large_key", &large_value).unwrap();

    let result = db.get_cf(nodes_cf, b"large_key").unwrap().unwrap();
    assert_eq!(result.len(), 1024 * 1024);
    assert_eq!(result[0], 0);
    assert_eq!(result[1024 * 1024 - 1], 255);

    println!("AFTER: Verified 1MB value storage");
}
