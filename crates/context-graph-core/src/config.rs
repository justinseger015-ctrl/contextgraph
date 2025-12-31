//! Configuration management for the Context Graph system.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{CoreError, CoreResult};

/// System development phase.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Ghost system phase - stubs and scaffolding
    #[default]
    Ghost,
    /// Development phase - active implementation
    Development,
    /// Production phase - fully operational
    Production,
}

/// Main configuration structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Current system phase
    #[serde(default)]
    pub phase: Phase,
    pub server: ServerConfig,
    pub mcp: McpConfig,
    pub logging: LoggingConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub index: IndexConfig,
    pub utl: UtlConfig,
    pub features: FeatureFlags,
    pub cuda: CudaConfig,
}

impl Config {
    /// Load configuration from files and environment.
    ///
    /// Configuration is loaded in order:
    /// 1. config/default.toml (base settings)
    /// 2. config/{CONTEXT_GRAPH_ENV}.toml (environment-specific)
    /// 3. Environment variables with CONTEXT_GRAPH_ prefix
    pub fn load() -> CoreResult<Self> {
        let env = std::env::var("CONTEXT_GRAPH_ENV").unwrap_or_else(|_| "development".to_string());

        let builder = config::Config::builder()
            .add_source(config::File::with_name("config/default").required(false))
            .add_source(config::File::with_name(&format!("config/{}", env)).required(false))
            .add_source(config::Environment::with_prefix("CONTEXT_GRAPH").separator("__"));

        let config: Config = builder.build()?.try_deserialize()?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration with defaults for testing/development.
    pub fn default_config() -> Self {
        Self {
            phase: Phase::default(),
            server: ServerConfig::default(),
            mcp: McpConfig::default(),
            logging: LoggingConfig::default(),
            storage: StorageConfig::default(),
            embedding: EmbeddingConfig::default(),
            index: IndexConfig::default(),
            utl: UtlConfig::default(),
            features: FeatureFlags::default(),
            cuda: CudaConfig::default(),
        }
    }

    /// Load configuration from a TOML file.
    pub fn from_file(path: &std::path::Path) -> CoreResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            CoreError::ConfigError(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;

        let config: Config = toml::from_str(&content)
            .map_err(|e| CoreError::ConfigError(format!("Failed to parse config file: {}", e)))?;

        config.validate()?;
        Ok(config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::default_config()
    }
}

impl Config {
    /// Validate configuration values.
    pub fn validate(&self) -> CoreResult<()> {
        if self.mcp.max_payload_size == 0 {
            return Err(CoreError::ConfigError(
                "mcp.max_payload_size must be greater than 0".into(),
            ));
        }

        if self.embedding.dimension == 0 {
            return Err(CoreError::ConfigError(
                "embedding.dimension must be greater than 0".into(),
            ));
        }

        if self.storage.backend != "memory" {
            let path = PathBuf::from(&self.storage.path);
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    return Err(CoreError::ConfigError(format!(
                        "storage.path parent directory does not exist: {}",
                        parent.display()
                    )));
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub name: String,
    pub version: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            name: "context-graph".to_string(),
            version: "0.1.0-ghost".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpConfig {
    pub transport: String,
    pub max_payload_size: usize,
    pub request_timeout: u64,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            transport: "stdio".to_string(),
            max_payload_size: 10_485_760, // 10MB
            request_timeout: 30,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub include_location: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            include_location: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    pub backend: String,
    pub path: String,
    pub compression: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),
            path: "./data/storage".to_string(),
            compression: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dimension: usize,
    pub max_input_length: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "stub".to_string(),
            dimension: 1536,
            max_input_length: 8191,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IndexConfig {
    pub backend: String,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UtlConfig {
    pub mode: String,
    pub default_emotional_weight: f32,
    pub consolidation_threshold: f32,
}

impl Default for UtlConfig {
    fn default() -> Self {
        Self {
            mode: "stub".to_string(),
            default_emotional_weight: 1.0,
            consolidation_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeatureFlags {
    pub utl_enabled: bool,
    pub dream_enabled: bool,
    pub neuromodulation_enabled: bool,
    pub active_inference_enabled: bool,
    pub immune_enabled: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            utl_enabled: true,
            dream_enabled: false,
            neuromodulation_enabled: false,
            active_inference_enabled: false,
            immune_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CudaConfig {
    pub enabled: bool,
    pub device_id: u32,
    pub memory_limit_gb: f32,
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: 0,
            memory_limit_gb: 4.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default_config();
        assert_eq!(config.server.name, "context-graph");
        assert_eq!(config.embedding.dimension, 1536);
        assert!(!config.cuda.enabled);
    }

    #[test]
    fn test_validation_passes() {
        let config = Config::default_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_fails_zero_payload() {
        let mut config = Config::default_config();
        config.mcp.max_payload_size = 0;
        assert!(config.validate().is_err());
    }

    // =========================================================================
    // TC-GHOST-006: Configuration & Infrastructure Tests
    // =========================================================================

    #[test]
    fn test_config_serialization_round_trip() {
        // TC-GHOST-006: Config must serialize and deserialize exactly
        let config = Config::default_config();

        // Serialize to TOML
        let toml_str = toml::to_string(&config).expect("Config must serialize to TOML");

        // Deserialize back
        let deserialized: Config =
            toml::from_str(&toml_str).expect("Config must deserialize from TOML");

        // Verify all fields match
        assert_eq!(deserialized.phase, config.phase, "Phase must match");
        assert_eq!(
            deserialized.server.name, config.server.name,
            "Server name must match"
        );
        assert_eq!(
            deserialized.server.version, config.server.version,
            "Server version must match"
        );
        assert_eq!(
            deserialized.mcp.transport, config.mcp.transport,
            "MCP transport must match"
        );
        assert_eq!(
            deserialized.mcp.max_payload_size, config.mcp.max_payload_size,
            "MCP max_payload_size must match"
        );
        assert_eq!(
            deserialized.logging.level, config.logging.level,
            "Logging level must match"
        );
        assert_eq!(
            deserialized.embedding.dimension, config.embedding.dimension,
            "Embedding dimension must match"
        );
    }

    #[test]
    fn test_config_serialization_json_round_trip() {
        // TC-GHOST-006: Config must also round-trip through JSON
        let config = Config::default_config();

        // Serialize to JSON
        let json_str = serde_json::to_string(&config).expect("Config must serialize to JSON");

        // Deserialize back
        let deserialized: Config =
            serde_json::from_str(&json_str).expect("Config must deserialize from JSON");

        // Verify critical fields
        assert_eq!(deserialized.phase, config.phase);
        assert_eq!(deserialized.embedding.dimension, config.embedding.dimension);
        assert_eq!(
            deserialized.features.dream_enabled,
            config.features.dream_enabled
        );
    }

    #[test]
    fn test_config_phase_serialization() {
        // TC-GHOST-006: Phase enum must serialize correctly
        let phases = [Phase::Ghost, Phase::Development, Phase::Production];
        let expected = ["ghost", "development", "production"];

        for (phase, expected_str) in phases.iter().zip(expected.iter()) {
            let json = serde_json::to_string(phase).expect("Phase must serialize");
            assert_eq!(
                json,
                format!("\"{}\"", expected_str),
                "Phase {:?} must serialize as {}",
                phase,
                expected_str
            );

            let deserialized: Phase = serde_json::from_str(&json).expect("Phase must deserialize");
            assert_eq!(&deserialized, phase, "Phase must round-trip correctly");
        }
    }

    #[test]
    fn test_config_validation_embedding_dimension_zero() {
        // TC-GHOST-006: Embedding dimension 0 must fail validation
        let mut config = Config::default_config();
        config.embedding.dimension = 0;

        let result = config.validate();
        assert!(
            result.is_err(),
            "Embedding dimension 0 must fail validation"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("embedding.dimension"),
            "Error must mention embedding.dimension"
        );
    }

    #[test]
    fn test_feature_flags_default_values() {
        // TC-GHOST-006: Feature flags must have correct defaults
        let features = FeatureFlags::default();

        // Ghost System phase: UTL enabled, everything else disabled
        assert!(features.utl_enabled, "UTL must be enabled by default");
        assert!(!features.dream_enabled, "Dream must be disabled by default");
        assert!(
            !features.neuromodulation_enabled,
            "Neuromodulation must be disabled by default"
        );
        assert!(
            !features.active_inference_enabled,
            "Active inference must be disabled by default"
        );
        assert!(
            !features.immune_enabled,
            "Immune system must be disabled by default"
        );
    }

    #[test]
    fn test_feature_flags_serialization() {
        // TC-GHOST-006: Feature flags must serialize/deserialize correctly
        let mut features = FeatureFlags::default();
        features.dream_enabled = true;
        features.neuromodulation_enabled = true;

        let json = serde_json::to_string(&features).expect("FeatureFlags must serialize");
        let deserialized: FeatureFlags =
            serde_json::from_str(&json).expect("FeatureFlags must deserialize");

        assert_eq!(deserialized.utl_enabled, features.utl_enabled);
        assert_eq!(deserialized.dream_enabled, features.dream_enabled);
        assert_eq!(
            deserialized.neuromodulation_enabled,
            features.neuromodulation_enabled
        );
        assert_eq!(
            deserialized.active_inference_enabled,
            features.active_inference_enabled
        );
        assert_eq!(deserialized.immune_enabled, features.immune_enabled);
    }

    #[test]
    fn test_server_config_defaults() {
        // TC-GHOST-006: Server config must have correct defaults
        let server = ServerConfig::default();

        assert_eq!(
            server.name, "context-graph",
            "Server name must be context-graph"
        );
        assert_eq!(
            server.version, "0.1.0-ghost",
            "Server version must be 0.1.0-ghost"
        );
    }

    #[test]
    fn test_mcp_config_defaults() {
        // TC-GHOST-006: MCP config must have correct defaults
        let mcp = McpConfig::default();

        assert_eq!(mcp.transport, "stdio", "Transport must be stdio");
        assert_eq!(mcp.max_payload_size, 10_485_760, "Max payload must be 10MB");
        assert_eq!(
            mcp.request_timeout, 30,
            "Request timeout must be 30 seconds"
        );
    }

    #[test]
    fn test_logging_config_defaults() {
        // TC-GHOST-006: Logging config must have correct defaults
        let logging = LoggingConfig::default();

        assert_eq!(logging.level, "info", "Default log level must be info");
        assert_eq!(
            logging.format, "pretty",
            "Default log format must be pretty"
        );
        assert!(
            !logging.include_location,
            "Location should be disabled by default"
        );
    }

    #[test]
    fn test_storage_config_defaults() {
        // TC-GHOST-006: Storage config must have correct defaults
        let storage = StorageConfig::default();

        assert_eq!(storage.backend, "memory", "Default backend must be memory");
        assert_eq!(
            storage.path, "./data/storage",
            "Default path must be ./data/storage"
        );
        assert!(
            storage.compression,
            "Compression should be enabled by default"
        );
    }

    #[test]
    fn test_embedding_config_defaults() {
        // TC-GHOST-006: Embedding config must have correct defaults
        let embedding = EmbeddingConfig::default();

        assert_eq!(embedding.model, "stub", "Default model must be stub");
        assert_eq!(
            embedding.dimension, 1536,
            "Dimension must be 1536 (OpenAI compatible)"
        );
        assert_eq!(
            embedding.max_input_length, 8191,
            "Max input length must be 8191"
        );
    }

    #[test]
    fn test_utl_config_defaults() {
        // TC-GHOST-006: UTL config must have correct defaults
        let utl = UtlConfig::default();

        assert_eq!(utl.mode, "stub", "Default mode must be stub");
        assert_eq!(
            utl.default_emotional_weight, 1.0,
            "Default emotional weight must be 1.0"
        );
        assert_eq!(
            utl.consolidation_threshold, 0.7,
            "Consolidation threshold must be 0.7"
        );
    }

    #[test]
    fn test_cuda_config_defaults() {
        // TC-GHOST-006: CUDA config must have correct defaults
        let cuda = CudaConfig::default();

        assert!(!cuda.enabled, "CUDA must be disabled by default");
        assert_eq!(cuda.device_id, 0, "Default device ID must be 0");
        assert_eq!(
            cuda.memory_limit_gb, 4.0,
            "Default memory limit must be 4GB"
        );
    }

    #[test]
    fn test_config_from_toml_string() {
        // TC-GHOST-006: Config must parse from minimal TOML
        let toml_str = r#"
            [server]
            name = "test-server"
            version = "1.0.0"

            [mcp]
            transport = "stdio"
            max_payload_size = 1000000
            request_timeout = 60

            [logging]
            level = "debug"
            format = "json"
            include_location = true

            [storage]
            backend = "memory"
            path = "/tmp/test"
            compression = false

            [embedding]
            model = "custom"
            dimension = 768
            max_input_length = 4096

            [index]
            backend = "memory"
            hnsw_m = 32
            hnsw_ef_construction = 400

            [utl]
            mode = "real"
            default_emotional_weight = 1.2
            consolidation_threshold = 0.8

            [features]
            utl_enabled = true
            dream_enabled = true
            neuromodulation_enabled = false
            active_inference_enabled = false
            immune_enabled = false

            [cuda]
            enabled = false
            device_id = 1
            memory_limit_gb = 8.0
        "#;

        let config: Config = toml::from_str(toml_str).expect("Config must parse from TOML");

        // Verify parsed values
        assert_eq!(config.server.name, "test-server");
        assert_eq!(config.server.version, "1.0.0");
        assert_eq!(config.mcp.max_payload_size, 1000000);
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.embedding.dimension, 768);
        assert_eq!(config.utl.consolidation_threshold, 0.8);
        assert!(config.features.dream_enabled);
        assert_eq!(config.cuda.device_id, 1);
    }

    #[test]
    fn test_config_full_structure_integrity() {
        // TC-GHOST-006: Full config structure must be maintained through serialization
        let mut config = Config::default_config();

        // Modify various fields
        config.server.name = "modified-server".to_string();
        config.mcp.request_timeout = 120;
        config.logging.level = "trace".to_string();
        config.embedding.dimension = 768;
        config.features.dream_enabled = true;
        config.cuda.enabled = true;
        config.cuda.memory_limit_gb = 16.0;

        // Round-trip through TOML
        let toml_str = toml::to_string(&config).unwrap();
        let restored: Config = toml::from_str(&toml_str).unwrap();

        // Verify all modifications survived
        assert_eq!(restored.server.name, "modified-server");
        assert_eq!(restored.mcp.request_timeout, 120);
        assert_eq!(restored.logging.level, "trace");
        assert_eq!(restored.embedding.dimension, 768);
        assert!(restored.features.dream_enabled);
        assert!(restored.cuda.enabled);
        assert_eq!(restored.cuda.memory_limit_gb, 16.0);
    }
}
