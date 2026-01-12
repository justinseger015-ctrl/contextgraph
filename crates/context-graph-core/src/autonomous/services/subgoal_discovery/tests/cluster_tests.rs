//! Tests for MemoryCluster.

use crate::autonomous::curation::MemoryId;

use super::super::cluster::MemoryCluster;
use super::{make_cluster, make_labeled_cluster};

#[test]
fn test_memory_cluster_new() {
    let centroid = vec![0.1, 0.2, 0.3];
    let members = vec![MemoryId::new(), MemoryId::new()];
    let cluster = MemoryCluster::new(centroid.clone(), members.clone(), 0.8);

    assert_eq!(cluster.centroid, centroid);
    assert_eq!(cluster.members.len(), 2);
    assert!((cluster.coherence - 0.8).abs() < f32::EPSILON);
    assert!(cluster.label.is_none());
    assert!((cluster.avg_alignment - 0.0).abs() < f32::EPSILON);

    println!("[PASS] test_memory_cluster_new");
}

#[test]
fn test_memory_cluster_with_label() {
    let cluster = make_cluster(5, 0.7, 0.5).with_label("Test Label");

    assert_eq!(cluster.label, Some("Test Label".to_string()));

    println!("[PASS] test_memory_cluster_with_label");
}

#[test]
fn test_memory_cluster_with_avg_alignment() {
    let cluster = make_cluster(5, 0.7, 0.0).with_avg_alignment(0.85);

    assert!((cluster.avg_alignment - 0.85).abs() < f32::EPSILON);

    println!("[PASS] test_memory_cluster_with_avg_alignment");
}

#[test]
fn test_memory_cluster_alignment_clamping() {
    let cluster1 = make_cluster(5, 0.7, 0.0).with_avg_alignment(1.5);
    assert!((cluster1.avg_alignment - 1.0).abs() < f32::EPSILON);

    let cluster2 = make_cluster(5, 0.7, 0.0).with_avg_alignment(-0.5);
    assert!((cluster2.avg_alignment - 0.0).abs() < f32::EPSILON);

    println!("[PASS] test_memory_cluster_alignment_clamping");
}

#[test]
fn test_memory_cluster_size() {
    let cluster = make_cluster(10, 0.7, 0.5);
    assert_eq!(cluster.size(), 10);

    println!("[PASS] test_memory_cluster_size");
}

#[test]
fn test_memory_cluster_is_empty() {
    let empty_cluster = MemoryCluster::new(vec![0.1], vec![], 0.5);
    assert!(empty_cluster.is_empty());

    let non_empty = make_cluster(5, 0.7, 0.5);
    assert!(!non_empty.is_empty());

    println!("[PASS] test_memory_cluster_is_empty");
}

#[test]
#[should_panic(expected = "Centroid cannot be empty")]
fn test_memory_cluster_empty_centroid_panics() {
    MemoryCluster::new(vec![], vec![MemoryId::new()], 0.5);
}

#[test]
#[should_panic(expected = "Coherence must be in")]
fn test_memory_cluster_invalid_coherence_panics() {
    MemoryCluster::new(vec![0.1], vec![], 1.5);
}

#[test]
fn test_memory_cluster_labeled() {
    let cluster = make_labeled_cluster(5, 0.7, 0.5, "Test");
    assert_eq!(cluster.label, Some("Test".to_string()));

    println!("[PASS] test_memory_cluster_labeled");
}
