//! Tests for ProfileManager
//!
//! Comprehensive test suite for ProfileManager functionality.

#[cfg(test)]
mod tests {
    use crate::teleological::services::profile_manager::{
        builtin, ProfileManager, ProfileManagerConfig, ProfileMatch, ProfileStats,
    };
    use crate::teleological::{GroupType, ProfileId, NUM_EMBEDDERS};

    // ===== ProfileManagerConfig Tests =====

    #[test]
    fn test_config_default() {
        let config = ProfileManagerConfig::default();

        assert_eq!(config.max_profiles, 100);
        assert!(config.auto_create);
        assert_eq!(config.default_profile_id, "code_implementation");

        println!("[PASS] ProfileManagerConfig::default has correct values");
    }

    // ===== Built-in Profile Tests =====

    #[test]
    fn test_code_implementation_profile() {
        let profile = ProfileManager::code_implementation();

        assert_eq!(profile.id.as_str(), "code_implementation");
        assert!(profile.is_system);

        // E6 (index 5) should have highest weight
        let max_idx = profile
            .embedding_weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 5, "E6 should have highest weight");

        // Weights should sum to 1.0 (normalized)
        let sum: f32 = profile.embedding_weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to ~1.0, got {}",
            sum
        );

        println!("[PASS] code_implementation profile emphasizes E6");
    }

    #[test]
    fn test_research_analysis_profile() {
        let profile = ProfileManager::research_analysis();

        assert_eq!(profile.id.as_str(), "research_analysis");
        assert!(profile.is_system);

        // E1 (index 0), E4 (index 3), E7 (index 6) should be high
        assert!(profile.embedding_weights[0] > 0.15, "E1 should be high");
        assert!(profile.embedding_weights[3] > 0.15, "E4 should be high");
        assert!(profile.embedding_weights[6] > 0.10, "E7 should be high");

        println!("[PASS] research_analysis profile emphasizes E1, E4, E7");
    }

    #[test]
    fn test_creative_writing_profile() {
        let profile = ProfileManager::creative_writing();

        assert_eq!(profile.id.as_str(), "creative_writing");
        assert!(profile.is_system);

        // E10 (index 9), E11 (index 10) should be high
        assert!(profile.embedding_weights[9] > 0.15, "E10 should be high");
        assert!(profile.embedding_weights[10] > 0.15, "E11 should be high");

        println!("[PASS] creative_writing profile emphasizes E10, E11");
    }

    #[test]
    fn test_builtin_module_functions() {
        // Test that builtin module functions work correctly
        let code = builtin::code_implementation();
        let research = builtin::research_analysis();
        let creative = builtin::creative_writing();

        assert_eq!(code.id.as_str(), "code_implementation");
        assert_eq!(research.id.as_str(), "research_analysis");
        assert_eq!(creative.id.as_str(), "creative_writing");

        println!("[PASS] builtin module functions work correctly");
    }

    // ===== ProfileManager Basic Tests =====

    #[test]
    fn test_profile_manager_new() {
        let manager = ProfileManager::new();

        // Should have 3 built-in profiles
        assert_eq!(manager.profile_count(), 3);

        // Should have code_implementation
        let code_id = ProfileId::new("code_implementation");
        assert!(manager.contains(&code_id));

        // Should have research_analysis
        let research_id = ProfileId::new("research_analysis");
        assert!(manager.contains(&research_id));

        // Should have creative_writing
        let creative_id = ProfileId::new("creative_writing");
        assert!(manager.contains(&creative_id));

        println!("[PASS] ProfileManager::new creates manager with built-in profiles");
    }

    #[test]
    fn test_profile_manager_with_config() {
        let config = ProfileManagerConfig {
            max_profiles: 50,
            auto_create: false,
            default_profile_id: "research_analysis".to_string(),
        };

        let manager = ProfileManager::with_config(config);

        assert_eq!(manager.config().max_profiles, 50);
        assert!(!manager.config().auto_create);
        assert_eq!(manager.config().default_profile_id, "research_analysis");

        println!("[PASS] ProfileManager::with_config uses custom config");
    }

    // ===== CRUD Tests =====

    #[test]
    fn test_create_profile() {
        let mut manager = ProfileManager::new();

        let weights = [0.1; NUM_EMBEDDERS];
        let profile = manager.create_profile("test_profile", weights);

        assert_eq!(profile.id.as_str(), "test_profile");
        assert_eq!(manager.profile_count(), 4); // 3 built-in + 1 new

        // Weights should be normalized
        let sum: f32 = profile.embedding_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        println!("[PASS] create_profile creates and normalizes profile");
    }

    #[test]
    fn test_get_profile() {
        let manager = ProfileManager::new();

        let code_id = ProfileId::new("code_implementation");
        let profile = manager.get_profile(&code_id);

        assert!(profile.is_some());
        assert_eq!(profile.unwrap().id, code_id);

        let non_existent = ProfileId::new("non_existent");
        assert!(manager.get_profile(&non_existent).is_none());

        println!("[PASS] get_profile returns correct results");
    }

    #[test]
    fn test_update_profile() {
        let mut manager = ProfileManager::new();

        let code_id = ProfileId::new("code_implementation");

        // Update with new weights
        let mut new_weights = [0.05; NUM_EMBEDDERS];
        new_weights[0] = 0.5; // Boost E1

        let updated = manager.update_profile(&code_id, new_weights);
        assert!(updated);

        // Verify update
        let profile = manager.get_profile(&code_id).unwrap();
        assert!(profile.embedding_weights[0] > 0.3); // Should be normalized but still high

        // Update non-existent profile
        let fake_id = ProfileId::new("fake");
        assert!(!manager.update_profile(&fake_id, [0.1; NUM_EMBEDDERS]));

        println!("[PASS] update_profile updates existing profiles");
    }

    #[test]
    fn test_delete_profile() {
        let mut manager = ProfileManager::new();

        let code_id = ProfileId::new("code_implementation");
        assert!(manager.contains(&code_id));

        let deleted = manager.delete_profile(&code_id);
        assert!(deleted);
        assert!(!manager.contains(&code_id));
        assert_eq!(manager.profile_count(), 2);

        // Delete non-existent
        let fake_id = ProfileId::new("fake");
        assert!(!manager.delete_profile(&fake_id));

        println!("[PASS] delete_profile removes profiles correctly");
    }

    // ===== Profile Matching Tests =====

    #[test]
    fn test_find_best_match_code() {
        let manager = ProfileManager::new();

        let result = manager.find_best_match("implement a sorting algorithm");
        assert!(result.is_some());

        let matched = result.unwrap();
        assert_eq!(matched.profile_id.as_str(), "code_implementation");
        assert!(matched.similarity > 0.1);
        assert!(
            matched.reason.contains("code")
                || matched.reason.contains("implement")
                || matched.reason.contains("algorithm")
        );

        println!("[PASS] find_best_match matches code context to code_implementation");
    }

    #[test]
    fn test_find_best_match_research() {
        let manager = ProfileManager::new();

        let result = manager.find_best_match("analyze the research and explain why this happens");
        assert!(result.is_some());

        let matched = result.unwrap();
        assert_eq!(matched.profile_id.as_str(), "research_analysis");
        assert!(matched.similarity > 0.1);

        println!("[PASS] find_best_match matches research context to research_analysis");
    }

    #[test]
    fn test_find_best_match_creative() {
        let manager = ProfileManager::new();

        let result = manager.find_best_match("write a creative story about imagination");
        assert!(result.is_some());

        let matched = result.unwrap();
        assert_eq!(matched.profile_id.as_str(), "creative_writing");
        assert!(matched.similarity > 0.1);

        println!("[PASS] find_best_match matches creative context to creative_writing");
    }

    #[test]
    fn test_find_best_match_default() {
        let manager = ProfileManager::new();

        // Context that doesn't match any keywords specifically
        let result = manager.find_best_match("hello world");
        assert!(result.is_some());

        let matched = result.unwrap();
        assert_eq!(matched.profile_id.as_str(), "code_implementation"); // Default
        assert!(matched.similarity < 0.5); // Low similarity

        println!("[PASS] find_best_match returns default for ambiguous context");
    }

    // ===== List Profiles Tests =====

    #[test]
    fn test_list_profiles() {
        let manager = ProfileManager::new();

        let ids = manager.list_profiles();
        assert_eq!(ids.len(), 3);

        let id_strs: Vec<&str> = ids.iter().map(|id| id.as_str()).collect();
        assert!(id_strs.contains(&"code_implementation"));
        assert!(id_strs.contains(&"research_analysis"));
        assert!(id_strs.contains(&"creative_writing"));

        println!("[PASS] list_profiles returns all profile IDs");
    }

    // ===== Stats Tests =====

    #[test]
    fn test_record_usage_and_get_stats() {
        let mut manager = ProfileManager::new();

        let code_id = ProfileId::new("code_implementation");

        // Initially no stats
        assert!(manager.get_stats(&code_id).is_none());

        // Record usage
        manager.record_usage(&code_id, 0.8);
        manager.record_usage(&code_id, 0.9);
        manager.record_usage(&code_id, 0.7);

        // Get stats
        let stats = manager.get_stats(&code_id);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.usage_count, 3);
        assert!((stats.avg_effectiveness - 0.8).abs() < 0.01); // (0.8 + 0.9 + 0.7) / 3 = 0.8
        assert!(stats.last_used > 0);

        println!("[PASS] record_usage and get_stats track usage correctly");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_record_usage_non_existent_panics() {
        let mut manager = ProfileManager::new();
        let fake_id = ProfileId::new("fake");
        manager.record_usage(&fake_id, 0.5);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_record_usage_invalid_effectiveness_panics() {
        let mut manager = ProfileManager::new();
        let code_id = ProfileId::new("code_implementation");
        manager.record_usage(&code_id, 1.5); // Out of range
    }

    // ===== Get or Create Default Tests =====

    #[test]
    fn test_get_or_create_default() {
        let mut manager = ProfileManager::new();

        let profile = manager.get_or_create_default();
        assert_eq!(profile.id.as_str(), "code_implementation");

        println!("[PASS] get_or_create_default returns default profile");
    }

    #[test]
    fn test_get_or_create_default_creates_if_missing() {
        let config = ProfileManagerConfig {
            max_profiles: 100,
            auto_create: true,
            default_profile_id: "code_implementation".to_string(),
        };

        let mut manager = ProfileManager::with_config(config);

        // Delete the default profile
        let code_id = ProfileId::new("code_implementation");
        manager.delete_profile(&code_id);
        assert!(!manager.contains(&code_id));

        // get_or_create_default should recreate it
        let profile = manager.get_or_create_default();
        assert_eq!(profile.id.as_str(), "code_implementation");
        assert!(manager.contains(&code_id));

        println!("[PASS] get_or_create_default recreates missing default profile");
    }

    // ===== Group Preference Tests =====

    #[test]
    fn test_get_profiles_for_group() {
        let manager = ProfileManager::new();

        // Implementation group should match code_implementation
        let impl_profiles = manager.get_profiles_for_group(GroupType::Implementation);
        assert!(!impl_profiles.is_empty());
        assert!(impl_profiles
            .iter()
            .any(|p| p.id.as_str() == "code_implementation"));

        // Qualitative group should match creative_writing
        let qual_profiles = manager.get_profiles_for_group(GroupType::Qualitative);
        assert!(!qual_profiles.is_empty());
        assert!(qual_profiles
            .iter()
            .any(|p| p.id.as_str() == "creative_writing"));

        println!("[PASS] get_profiles_for_group returns profiles with high group weights");
    }

    // ===== FAIL FAST Tests =====

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_create_profile_empty_id_panics() {
        let mut manager = ProfileManager::new();
        manager.create_profile("", [0.1; NUM_EMBEDDERS]);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_create_profile_negative_weight_panics() {
        let mut manager = ProfileManager::new();
        let mut weights = [0.1; NUM_EMBEDDERS];
        weights[0] = -0.1;
        manager.create_profile("test", weights);
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_update_profile_negative_weight_panics() {
        let mut manager = ProfileManager::new();
        let code_id = ProfileId::new("code_implementation");
        let mut weights = [0.1; NUM_EMBEDDERS];
        weights[5] = -0.5;
        manager.update_profile(&code_id, weights);
    }

    #[test]
    fn test_max_profiles_limit() {
        let config = ProfileManagerConfig {
            max_profiles: 5, // 3 built-in + 2 custom max
            auto_create: true,
            default_profile_id: "code_implementation".to_string(),
        };

        let mut manager = ProfileManager::with_config(config);
        assert_eq!(manager.profile_count(), 3); // Built-in profiles

        // Can add 2 more
        manager.create_profile("custom1", [0.1; NUM_EMBEDDERS]);
        manager.create_profile("custom2", [0.1; NUM_EMBEDDERS]);
        assert_eq!(manager.profile_count(), 5);

        println!("[PASS] Profile limit works correctly");
    }

    #[test]
    #[should_panic(expected = "FAIL FAST")]
    fn test_max_profiles_limit_exceeded_panics() {
        let config = ProfileManagerConfig {
            max_profiles: 3, // Only built-in profiles allowed
            auto_create: true,
            default_profile_id: "code_implementation".to_string(),
        };

        let mut manager = ProfileManager::with_config(config);
        manager.create_profile("extra", [0.1; NUM_EMBEDDERS]); // Should panic
    }

    // ===== Serialization Tests =====

    #[test]
    fn test_profile_stats_fields() {
        let stats = ProfileStats {
            profile_id: ProfileId::new("test"),
            usage_count: 10,
            avg_effectiveness: 0.85,
            last_used: 1234567890,
        };

        assert_eq!(stats.profile_id.as_str(), "test");
        assert_eq!(stats.usage_count, 10);
        assert!((stats.avg_effectiveness - 0.85).abs() < f32::EPSILON);
        assert_eq!(stats.last_used, 1234567890);

        println!("[PASS] ProfileStats has correct fields");
    }

    #[test]
    fn test_profile_match_fields() {
        let pm = ProfileMatch {
            profile_id: ProfileId::new("test"),
            similarity: 0.9,
            reason: "Test match".to_string(),
        };

        assert_eq!(pm.profile_id.as_str(), "test");
        assert!((pm.similarity - 0.9).abs() < f32::EPSILON);
        assert_eq!(pm.reason, "Test match");

        println!("[PASS] ProfileMatch has correct fields");
    }

    // ===== Default Trait Test =====

    #[test]
    fn test_profile_manager_default() {
        let manager = ProfileManager::default();
        assert_eq!(manager.profile_count(), 3);

        println!("[PASS] ProfileManager::default works");
    }
}
