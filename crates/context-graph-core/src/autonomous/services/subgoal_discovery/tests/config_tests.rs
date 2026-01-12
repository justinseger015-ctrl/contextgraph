//! Tests for DiscoveryConfig.

use crate::autonomous::evolution::GoalEvolutionConfig;

use super::super::config::DiscoveryConfig;

#[test]
fn test_discovery_config_default() {
    let config = DiscoveryConfig::default();

    assert_eq!(config.min_cluster_size, 10);
    assert!((config.min_coherence - 0.6).abs() < f32::EPSILON);
    assert!((config.emergence_threshold - 0.7).abs() < f32::EPSILON);
    assert_eq!(config.max_candidates, 20);
    assert!((config.min_confidence - 0.5).abs() < f32::EPSILON);

    println!("[PASS] test_discovery_config_default");
}

#[test]
fn test_discovery_config_from_evolution_config() {
    let evolution_config = GoalEvolutionConfig {
        min_cluster_size: 15,
        ..Default::default()
    };

    let discovery_config = DiscoveryConfig::from(&evolution_config);

    assert_eq!(discovery_config.min_cluster_size, 15);

    println!("[PASS] test_discovery_config_from_evolution_config");
}
