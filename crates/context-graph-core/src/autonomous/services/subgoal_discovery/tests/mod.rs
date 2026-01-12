//! Tests for sub-goal discovery module.

mod cluster_tests;
mod config_tests;
mod integration_tests;
mod result_tests;
mod service_tests;

use crate::autonomous::curation::MemoryId;

use super::cluster::MemoryCluster;

/// Helper to create a test cluster
pub(crate) fn make_cluster(size: usize, coherence: f32, alignment: f32) -> MemoryCluster {
    let members: Vec<MemoryId> = (0..size).map(|_| MemoryId::new()).collect();
    MemoryCluster::new(vec![0.1, 0.2, 0.3], members, coherence).with_avg_alignment(alignment)
}

/// Helper to create a labeled test cluster
pub(crate) fn make_labeled_cluster(
    size: usize,
    coherence: f32,
    alignment: f32,
    label: &str,
) -> MemoryCluster {
    make_cluster(size, coherence, alignment).with_label(label)
}
