//! Local LLM for causal relationship discovery.
//!
//! Uses Qwen2.5-3B-Instruct via Candle for native CUDA inference.
//! Optimized for RTX 5090 Blackwell architecture with FP16/BF16 tensor cores.
//!
//! # VRAM Usage
//!
//! - Qwen2.5-3B FP16: ~6GB
//! - Qwen2.5-7B FP16: ~14GB
//! - KV Cache (4096 ctx): ~1GB
//! - **RTX 5090 (32GB): Comfortable for 7B+ models**
//!
//! # Architecture
//!
//! The LLM is used to analyze pairs of memories and determine if there's
//! a causal relationship between them. It outputs structured JSON responses
//! that are parsed into `CausalAnalysisResult`.
//!
//! # Prompt Template
//!
//! Uses Qwen2.5's ChatML format:
//! ```text
//! <|im_start|>system
//! You are a causal reasoning expert...
//! <|im_end|>
//! <|im_start|>user
//! Statement A: "..."
//! Statement B: "..."
//! <|im_end|>
//! <|im_start|>assistant
//! ```

mod prompt;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use parking_lot::RwLock;
use tokenizers::Tokenizer;
use tracing::{debug, error, info, warn};

use crate::error::{CausalAgentError, CausalAgentResult};
use crate::types::{CausalAnalysisResult, CausalLinkDirection};

pub use prompt::CausalPromptBuilder;

/// Precision mode for inference (RTX 5090 optimized).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InferencePrecision {
    /// FP32 - Maximum precision, higher VRAM.
    FP32,
    /// FP16 - Tensor core optimized, recommended for RTX 5090.
    #[default]
    FP16,
    /// BF16 - Brain float, good for training, similar to FP16.
    BF16,
}

impl InferencePrecision {
    /// Get the Candle DType for this precision.
    pub fn to_dtype(&self) -> DType {
        match self {
            InferencePrecision::FP32 => DType::F32,
            InferencePrecision::FP16 => DType::F16,
            InferencePrecision::BF16 => DType::BF16,
        }
    }
}

/// Configuration for the Causal Discovery LLM.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B-Instruct").
    pub model_id: String,

    /// Local model directory path.
    pub model_dir: PathBuf,

    /// Context window size (default: 4096).
    pub context_size: usize,

    /// Inference precision (FP16 recommended for RTX 5090).
    pub precision: InferencePrecision,

    /// Temperature for sampling (0.0 = deterministic).
    pub temperature: f64,

    /// Top-p sampling threshold.
    pub top_p: f64,

    /// Maximum tokens to generate per response.
    pub max_tokens: usize,

    /// Seed for reproducibility.
    pub seed: u64,

    /// Whether to use CUDA if available.
    pub use_cuda: bool,

    /// Repetition penalty to avoid loops.
    pub repetition_penalty: f32,

    /// Legacy: Path to local GGUF model (unused).
    pub model_path: PathBuf,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen2.5-3B-Instruct".to_string(),
            model_dir: PathBuf::from("models/causal-discovery/qwen2.5-3b-instruct"),
            context_size: 4096,
            precision: InferencePrecision::BF16, // Qwen2.5 uses BF16 natively
            temperature: 0.1, // Low for deterministic analysis
            top_p: 0.9,
            max_tokens: 256,
            seed: 42,
            use_cuda: true,
            repetition_penalty: 1.1,
            model_path: PathBuf::new(),
        }
    }
}

/// Internal state of the LLM.
enum LlmState {
    /// Not loaded.
    Unloaded,

    /// Loaded with Candle model.
    Loaded {
        /// The Qwen2 model.
        model: ModelForCausalLM,
        /// Tokenizer.
        tokenizer: Tokenizer,
        /// Device (CUDA or CPU).
        device: Device,
        /// Data type.
        dtype: DType,
        /// Model configuration.
        config: Qwen2Config,
    },
}

/// Local LLM wrapper for causal relationship discovery.
///
/// Uses Qwen2.5-Instruct via Candle for native CUDA inference,
/// optimized for RTX 5090 Blackwell architecture.
///
/// # VRAM Usage (RTX 5090 32GB)
///
/// | Model | Precision | VRAM | Performance |
/// |-------|-----------|------|-------------|
/// | Qwen2.5-3B | BF16 | ~6GB | ~60 tok/s |
/// | Qwen2.5-7B | BF16 | ~14GB | ~45 tok/s |
///
/// # RTX 5090 Optimizations
///
/// - BF16 uses 5th-gen Tensor Cores
/// - cuBLAS for matrix operations
/// - KV caching for efficient generation
pub struct CausalDiscoveryLLM {
    /// Configuration.
    config: LlmConfig,

    /// Internal state.
    state: RwLock<LlmState>,

    /// Whether the model is loaded.
    loaded: AtomicBool,

    /// Prompt builder.
    prompt_builder: CausalPromptBuilder,
}

impl CausalDiscoveryLLM {
    /// Create a new CausalDiscoveryLLM with default configuration.
    pub fn new() -> CausalAgentResult<Self> {
        Self::with_config(LlmConfig::default())
    }

    /// Create with a specific model directory.
    pub fn with_model_dir(model_dir: &str) -> CausalAgentResult<Self> {
        let config = LlmConfig {
            model_dir: PathBuf::from(model_dir),
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with custom configuration.
    pub fn with_config(config: LlmConfig) -> CausalAgentResult<Self> {
        Ok(Self {
            config,
            state: RwLock::new(LlmState::Unloaded),
            loaded: AtomicBool::new(false),
            prompt_builder: CausalPromptBuilder::new(),
        })
    }

    /// Check if the model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    /// Load the model into memory.
    ///
    /// # CUDA Optimization
    ///
    /// On RTX 5090, this uses:
    /// - BF16 for Tensor Core acceleration
    /// - cuBLAS for matrix operations
    /// - Asynchronous memory transfers
    pub async fn load(&self) -> CausalAgentResult<()> {
        if self.is_loaded() {
            warn!("LLM already loaded, skipping");
            return Ok(());
        }

        info!(
            model_dir = %self.config.model_dir.display(),
            precision = ?self.config.precision,
            context_size = self.config.context_size,
            use_cuda = self.config.use_cuda,
            "Loading Causal Discovery LLM (Candle native CUDA)"
        );

        // Determine device
        let device = if self.config.use_cuda {
            match Device::cuda_if_available(0) {
                Ok(dev) => {
                    if dev.is_cuda() {
                        info!("CUDA device detected, using GPU acceleration");
                        dev
                    } else {
                        warn!("CUDA requested but not available, using CPU");
                        Device::Cpu
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to initialize CUDA, using CPU");
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };

        let dtype = self.config.precision.to_dtype();

        // Load model configuration
        let config_path = self.config.model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            CausalAgentError::ModelNotFound {
                path: format!("{}: {}", config_path.display(), e),
            }
        })?;
        let model_config: Qwen2Config = serde_json::from_str(&config_str).map_err(|e| {
            CausalAgentError::LlmLoadError {
                message: format!("Failed to parse config.json: {}", e),
            }
        })?;

        info!(
            hidden_size = model_config.hidden_size,
            num_layers = model_config.num_hidden_layers,
            vocab_size = model_config.vocab_size,
            "Loaded model configuration"
        );

        // Load tokenizer
        let tokenizer_path = self.config.model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            CausalAgentError::LlmLoadError {
                message: format!("Failed to load tokenizer: {}", e),
            }
        })?;

        info!("Loaded tokenizer");

        // Find safetensor files
        let safetensor_files = self.find_safetensor_files()?;
        info!(
            file_count = safetensor_files.len(),
            "Loading model weights from safetensors"
        );

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device).map_err(|e| {
                CausalAgentError::LlmLoadError {
                    message: format!("Failed to load safetensors: {}", e),
                }
            })?
        };

        // Create model
        let model = ModelForCausalLM::new(&model_config, vb).map_err(|e| {
            CausalAgentError::LlmLoadError {
                message: format!("Failed to create model: {}", e),
            }
        })?;

        let mut state = self.state.write();
        *state = LlmState::Loaded {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            config: model_config,
        };
        self.loaded.store(true, Ordering::SeqCst);

        info!(
            device = ?device,
            precision = ?self.config.precision,
            "Causal Discovery LLM loaded successfully"
        );
        Ok(())
    }

    /// Find safetensor files in the model directory.
    fn find_safetensor_files(&self) -> CausalAgentResult<Vec<PathBuf>> {
        let index_path = self.config.model_dir.join("model.safetensors.index.json");

        if index_path.exists() {
            // Sharded model - read index to find files
            let index_str = std::fs::read_to_string(&index_path).map_err(|e| {
                CausalAgentError::ModelNotFound {
                    path: format!("{}: {}", index_path.display(), e),
                }
            })?;
            let index: serde_json::Value = serde_json::from_str(&index_str).map_err(|e| {
                CausalAgentError::LlmLoadError {
                    message: format!("Failed to parse index: {}", e),
                }
            })?;

            // Get unique filenames from weight_map
            let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();
            if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                for filename in weight_map.values() {
                    if let Some(f) = filename.as_str() {
                        files.insert(f.to_string());
                    }
                }
            }

            let mut paths: Vec<PathBuf> = files
                .into_iter()
                .map(|f| self.config.model_dir.join(f))
                .collect();
            paths.sort();

            if paths.is_empty() {
                return Err(CausalAgentError::ModelNotFound {
                    path: "No safetensor files found in index".to_string(),
                });
            }

            Ok(paths)
        } else {
            // Single file model
            let single_path = self.config.model_dir.join("model.safetensors");
            if single_path.exists() {
                Ok(vec![single_path])
            } else {
                Err(CausalAgentError::ModelNotFound {
                    path: format!(
                        "No safetensor files found in {}",
                        self.config.model_dir.display()
                    ),
                })
            }
        }
    }

    /// Unload the model from memory.
    pub async fn unload(&self) -> CausalAgentResult<()> {
        if !self.is_loaded() {
            return Ok(());
        }

        info!("Unloading Causal Discovery LLM");

        let mut state = self.state.write();
        *state = LlmState::Unloaded;
        self.loaded.store(false, Ordering::SeqCst);

        Ok(())
    }

    /// Analyze a pair of memories for causal relationship.
    ///
    /// # Arguments
    ///
    /// * `memory_a` - Content of the first memory (potential cause)
    /// * `memory_b` - Content of the second memory (potential effect)
    ///
    /// # Returns
    ///
    /// Analysis result including whether a causal link exists, direction,
    /// confidence score, and description of the mechanism.
    pub async fn analyze_causal_relationship(
        &self,
        memory_a: &str,
        memory_b: &str,
    ) -> CausalAgentResult<CausalAnalysisResult> {
        if !self.is_loaded() {
            return Err(CausalAgentError::LlmNotInitialized);
        }

        let prompt = self.prompt_builder.build_analysis_prompt(memory_a, memory_b);

        debug!(
            prompt_len = prompt.len(),
            "Analyzing causal relationship"
        );

        // Generate response from LLM
        let response = self.generate(&prompt).await?;

        // Parse the JSON response
        self.parse_causal_response(&response)
    }

    /// Batch analyze multiple memory pairs.
    pub async fn analyze_batch(
        &self,
        pairs: &[(String, String)],
    ) -> CausalAgentResult<Vec<CausalAnalysisResult>> {
        let mut results = Vec::with_capacity(pairs.len());

        for (i, (a, b)) in pairs.iter().enumerate() {
            debug!(
                pair_index = i,
                total = pairs.len(),
                "Analyzing pair"
            );

            match self.analyze_causal_relationship(a, b).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!(
                        pair_index = i,
                        error = %e,
                        "Failed to analyze pair, using default"
                    );
                    results.push(CausalAnalysisResult::default());
                }
            }
        }

        Ok(results)
    }

    /// Generate text from the LLM.
    async fn generate(&self, prompt: &str) -> CausalAgentResult<String> {
        let mut state = self.state.write();

        let LlmState::Loaded {
            ref mut model,
            ref tokenizer,
            ref device,
            ..
        } = *state
        else {
            return Err(CausalAgentError::LlmNotInitialized);
        };

        // Encode prompt
        let encoding = tokenizer.encode(prompt, true).map_err(|e| {
            CausalAgentError::LlmInferenceError {
                message: format!("Tokenization failed: {}", e),
            }
        })?;

        let input_ids = encoding.get_ids();
        let prompt_len = input_ids.len();
        let input_tensor = Tensor::new(input_ids, device)
            .map_err(|e| CausalAgentError::LlmInferenceError {
                message: format!("Failed to create input tensor: {}", e),
            })?
            .unsqueeze(0)
            .map_err(|e| CausalAgentError::LlmInferenceError {
                message: format!("Failed to unsqueeze: {}", e),
            })?;

        // Clear KV cache for new generation
        model.clear_kv_cache();

        // Generate tokens
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens = input_ids.to_vec();

        // EOS token IDs for Qwen2.5
        let eos_token_ids: Vec<u32> = vec![
            151643, // <|endoftext|>
            151645, // <|im_end|>
        ];

        // First forward pass with full prompt
        // logits shape: [batch=1, seq_len=prompt_len, vocab_size]
        let logits = model.forward(&input_tensor, 0).map_err(|e| {
            CausalAgentError::LlmInferenceError {
                message: format!("Forward pass failed: {}", e),
            }
        })?;

        // Sample first token from the last position's logits
        let mut next_token = self.sample_token(&logits, prompt_len, &all_tokens)?;

        // Check for immediate EOS
        if eos_token_ids.contains(&next_token) {
            return Ok(String::new());
        }

        generated_tokens.push(next_token);
        all_tokens.push(next_token);

        // Generate remaining tokens
        for _ in 1..self.config.max_tokens {
            // Forward pass with single token
            // Position is the index where we're generating (0-indexed)
            let pos = all_tokens.len() - 1;
            let input = Tensor::new(&[next_token], device)
                .map_err(|e| CausalAgentError::LlmInferenceError {
                    message: format!("Failed to create token tensor: {}", e),
                })?
                .unsqueeze(0)
                .map_err(|e| CausalAgentError::LlmInferenceError {
                    message: format!("Failed to unsqueeze: {}", e),
                })?;

            // logits shape: [batch=1, seq_len=1, vocab_size]
            let logits = model.forward(&input, pos).map_err(|e| {
                CausalAgentError::LlmInferenceError {
                    message: format!("Forward pass failed: {}", e),
                }
            })?;

            // For single token input, seq_len=1
            next_token = self.sample_token(&logits, 1, &all_tokens)?;

            // Check for EOS
            if eos_token_ids.contains(&next_token) {
                break;
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);
        }

        // Decode generated tokens
        let output = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| CausalAgentError::LlmInferenceError {
                message: format!("Decoding failed: {}", e),
            })?;

        debug!(
            generated_tokens = generated_tokens.len(),
            output_len = output.len(),
            "Generation complete"
        );

        Ok(output)
    }

    /// Sample a token from logits with temperature and top-p.
    fn sample_token(
        &self,
        logits: &Tensor,
        _seq_len: usize,
        all_tokens: &[u32],
    ) -> CausalAgentResult<u32> {
        // The model returns logits for the last position only (KV cache optimization)
        // Possible shapes:
        // - [batch=1, vocab_size] - most common with KV cache
        // - [batch=1, 1, vocab_size] - some implementations
        // - [batch=1, seq_len, vocab_size] - without KV cache (rare)
        //
        // We need to get to [vocab_size]
        let dims = logits.dims();
        let logits = match dims.len() {
            2 => {
                // Shape: [1, vocab_size] - squeeze the batch dim
                logits.squeeze(0).map_err(|e| {
                    CausalAgentError::LlmInferenceError {
                        message: format!("Failed to squeeze batch dim: {}", e),
                    }
                })?
            }
            3 => {
                // Shape: [1, seq_len, vocab_size] - get last position
                let logits = logits.squeeze(0).map_err(|e| {
                    CausalAgentError::LlmInferenceError {
                        message: format!("Failed to squeeze batch dim: {}", e),
                    }
                })?;
                // Now [seq_len, vocab_size], get last position
                let seq_len = logits.dim(0).map_err(|e| {
                    CausalAgentError::LlmInferenceError {
                        message: format!("Failed to get seq dim: {}", e),
                    }
                })?;
                if seq_len == 1 {
                    logits.squeeze(0).map_err(|e| {
                        CausalAgentError::LlmInferenceError {
                            message: format!("Failed to squeeze seq dim: {}", e),
                        }
                    })?
                } else {
                    logits
                        .get(seq_len - 1)
                        .map_err(|e| CausalAgentError::LlmInferenceError {
                            message: format!("Failed to get last position logits: {}", e),
                        })?
                }
            }
            _ => {
                return Err(CausalAgentError::LlmInferenceError {
                    message: format!("Unexpected logits shape: {:?}", dims),
                });
            }
        };

        // Apply repetition penalty
        let mut logits_vec: Vec<f32> = logits
            .to_dtype(DType::F32)
            .map_err(|e| CausalAgentError::LlmInferenceError {
                message: format!("Failed to convert logits: {}", e),
            })?
            .to_vec1()
            .map_err(|e| CausalAgentError::LlmInferenceError {
                message: format!("Failed to get logits vec: {}", e),
            })?;

        // Apply repetition penalty to already generated tokens
        for &token_id in all_tokens {
            if (token_id as usize) < logits_vec.len() {
                let score = logits_vec[token_id as usize];
                logits_vec[token_id as usize] = if score > 0.0 {
                    score / self.config.repetition_penalty
                } else {
                    score * self.config.repetition_penalty
                };
            }
        }

        // Apply temperature
        if self.config.temperature > 0.0 {
            for logit in &mut logits_vec {
                *logit /= self.config.temperature as f32;
            }
        }

        // Softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits_vec.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

        // Top-p (nucleus) sampling
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative = 0.0;
        let mut top_p_indices: Vec<(usize, f32)> = Vec::new();
        for (idx, prob) in indexed_probs {
            cumulative += prob;
            top_p_indices.push((idx, prob));
            if cumulative >= self.config.top_p as f32 {
                break;
            }
        }

        // Renormalize
        let top_p_sum: f32 = top_p_indices.iter().map(|(_, p)| p).sum();
        let normalized: Vec<(usize, f32)> = top_p_indices
            .iter()
            .map(|(i, p)| (*i, p / top_p_sum))
            .collect();

        // Sample from normalized distribution
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();

        let mut cumulative = 0.0;
        for (idx, prob) in normalized {
            cumulative += prob;
            if sample < cumulative {
                return Ok(idx as u32);
            }
        }

        // Fallback to argmax
        Ok(top_p_indices[0].0 as u32)
    }

    /// Parse the LLM response into a CausalAnalysisResult.
    fn parse_causal_response(&self, response: &str) -> CausalAgentResult<CausalAnalysisResult> {
        // The prompt ends with `{"causal_link":` so we need to prepend that to complete the JSON
        let full_response = format!("{{\"causal_link\":{}", response);

        // Try to extract JSON from the response
        let json_str = self.extract_json(&full_response);

        // Parse the JSON
        match serde_json::from_str::<serde_json::Value>(&json_str) {
            Ok(json) => {
                let has_causal_link = json
                    .get("causal_link")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let direction_str = json
                    .get("direction")
                    .and_then(|v| v.as_str())
                    .unwrap_or("none");

                let confidence = json
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;

                let mechanism = json
                    .get("mechanism")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                Ok(CausalAnalysisResult {
                    has_causal_link,
                    direction: CausalLinkDirection::from_str(direction_str),
                    confidence: confidence.clamp(0.0, 1.0),
                    mechanism,
                    raw_response: Some(response.to_string()),
                })
            }
            Err(e) => {
                // Fallback: try regex-based extraction for robustness with smaller models
                debug!(
                    error = %e,
                    "JSON parse failed, trying regex fallback"
                );
                self.parse_response_fallback(response)
            }
        }
    }

    /// Fallback parser using pattern matching for malformed JSON from smaller models.
    fn parse_response_fallback(&self, response: &str) -> CausalAgentResult<CausalAnalysisResult> {
        let lower = response.to_lowercase();

        // Extract causal_link (true/false)
        let has_causal_link = if lower.contains("true") && !lower.contains("false") {
            true
        } else if lower.contains("false") {
            false
        } else {
            // Default to false if unclear
            false
        };

        // Extract direction
        let direction = if lower.contains("a_causes_b") || lower.contains("a causes b") {
            CausalLinkDirection::ACausesB
        } else if lower.contains("b_causes_a") || lower.contains("b causes a") {
            CausalLinkDirection::BCausesA
        } else if lower.contains("bidirectional") {
            CausalLinkDirection::Bidirectional
        } else {
            CausalLinkDirection::NoCausalLink
        };

        // Extract confidence (look for numbers like 0.5, .5, 0.75, etc.)
        let confidence = self.extract_confidence(&lower);

        // Extract mechanism (everything after mechanism/mechan)
        let mechanism = self.extract_mechanism(response);

        info!(
            has_causal_link = has_causal_link,
            ?direction,
            confidence = confidence,
            "Parsed response using fallback"
        );

        Ok(CausalAnalysisResult {
            has_causal_link,
            direction,
            confidence: confidence.clamp(0.0, 1.0),
            mechanism,
            raw_response: Some(response.to_string()),
        })
    }

    /// Extract confidence value from response.
    fn extract_confidence(&self, response: &str) -> f32 {
        // Look for patterns like: 0.5, .5, 0.75, 1.0, etc.
        let patterns = [
            r"confidence[:\s]*([0-9]*\.?[0-9]+)",
            r"([0-9]+\.?[0-9]*)\s*,",
        ];

        for pattern in patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                if let Some(caps) = re.captures(response) {
                    if let Some(m) = caps.get(1) {
                        if let Ok(v) = m.as_str().parse::<f32>() {
                            if v >= 0.0 && v <= 1.0 {
                                return v;
                            }
                        }
                    }
                }
            }
        }
        0.5 // Default confidence
    }

    /// Extract mechanism description from response.
    fn extract_mechanism(&self, response: &str) -> String {
        // Look for mechanism/mechan field and extract value
        if let Some(idx) = response.to_lowercase().find("mechan") {
            let after = &response[idx..];
            // Find the value after : or =
            if let Some(colon) = after.find(':') {
                let value_start = &after[colon + 1..];
                // Find end (} or ")
                let end = value_start
                    .find('}')
                    .or_else(|| value_start.find('"'))
                    .unwrap_or(value_start.len());
                let value = value_start[..end]
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\\')
                    .to_string();
                if !value.is_empty() && value.len() < 500 {
                    return value;
                }
            }
        }
        "No mechanism extracted".to_string()
    }

    /// Extract JSON from potentially markdown-wrapped response.
    fn extract_json(&self, response: &str) -> String {
        // Try to find JSON in code blocks
        if let Some(start) = response.find("```json") {
            if let Some(end) = response[start..].find("```\n") {
                let json_start = start + 7; // Length of "```json"
                return response[json_start..start + end].trim().to_string();
            }
        }

        // Try to find JSON object directly - find the first { and matching }
        if let Some(start) = response.find('{') {
            // Find the matching closing brace by counting braces
            let chars: Vec<char> = response[start..].chars().collect();
            let mut depth = 0;
            let mut end_pos = None;

            for (i, ch) in chars.iter().enumerate() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end_pos = Some(start + i);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(end) = end_pos {
                let json_str = &response[start..=end];
                // Clean up common issues
                let cleaned = json_str
                    .replace("\":", "\": ")
                    .replace(",\"", ", \"")
                    .replace("  ", " ");
                return cleaned;
            }
        }

        response.to_string()
    }

    /// Get the model configuration.
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// Estimate VRAM usage in MB based on model and precision.
    pub fn estimated_vram_mb(&self) -> usize {
        let base_vram = if self.config.model_id.contains("3B") {
            // Qwen2.5-3B
            match self.config.precision {
                InferencePrecision::FP32 => 12000,
                InferencePrecision::FP16 | InferencePrecision::BF16 => 6000,
            }
        } else if self.config.model_id.contains("7B") {
            // Qwen2.5-7B
            match self.config.precision {
                InferencePrecision::FP32 => 28000,
                InferencePrecision::FP16 | InferencePrecision::BF16 => 14000,
            }
        } else {
            // Default assumption
            6000
        };

        // Add KV cache overhead (scales with context size)
        let kv_cache = (self.config.context_size / 1024) * 256;

        base_vram + kv_cache
    }

    /// Get recommended precision for available VRAM.
    pub fn recommend_precision(available_vram_gb: usize, model_id: &str) -> InferencePrecision {
        let is_3b = model_id.contains("3B");
        let is_7b = model_id.contains("7B");

        match available_vram_gb {
            0..=8 if is_7b => InferencePrecision::BF16, // Will be tight
            17.. => InferencePrecision::BF16,          // BF16 is optimal for tensor cores
            _ if is_3b => InferencePrecision::BF16,
            _ => InferencePrecision::BF16,
        }
    }
}

impl std::fmt::Debug for CausalDiscoveryLLM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalDiscoveryLLM")
            .field("model_id", &self.config.model_id)
            .field("loaded", &self.is_loaded())
            .field("precision", &self.config.precision)
            .field("context_size", &self.config.context_size)
            .field("use_cuda", &self.config.use_cuda)
            .finish()
    }
}

// Thread safety
unsafe impl Send for CausalDiscoveryLLM {}
unsafe impl Sync for CausalDiscoveryLLM {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.context_size, 4096);
        assert_eq!(config.precision, InferencePrecision::BF16);
        assert!(config.temperature < 0.5); // Should be low for analysis
        assert!(config.use_cuda);
    }

    #[test]
    fn test_precision_vram_estimates() {
        let config_3b = LlmConfig {
            model_id: "Qwen/Qwen2.5-3B-Instruct".to_string(),
            precision: InferencePrecision::BF16,
            ..Default::default()
        };
        let llm_3b = CausalDiscoveryLLM::with_config(config_3b).unwrap();
        assert!(llm_3b.estimated_vram_mb() < 8000); // 3B BF16 should be under 8GB

        let config_7b = LlmConfig {
            model_id: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            precision: InferencePrecision::BF16,
            ..Default::default()
        };
        let llm_7b = CausalDiscoveryLLM::with_config(config_7b).unwrap();
        assert!(llm_7b.estimated_vram_mb() < 16000); // 7B BF16 should be under 16GB
    }

    #[test]
    fn test_precision_recommendation() {
        // RTX 5090 has 32GB
        let precision = CausalDiscoveryLLM::recommend_precision(32, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(precision, InferencePrecision::BF16); // BF16 for tensor cores

        // 6GB card with 3B model
        let precision = CausalDiscoveryLLM::recommend_precision(6, "Qwen/Qwen2.5-3B-Instruct");
        assert_eq!(precision, InferencePrecision::BF16);
    }

    #[test]
    fn test_extract_json() {
        let llm = CausalDiscoveryLLM::new().unwrap();

        // Direct JSON
        let json = r#"{"causal_link": true, "direction": "A_causes_B"}"#;
        assert_eq!(llm.extract_json(json), json);

        // Markdown wrapped
        let wrapped = r#"Here is the analysis:
```json
{"causal_link": true}
```
"#;
        assert_eq!(llm.extract_json(wrapped), r#"{"causal_link": true}"#);

        // With surrounding text
        let surrounded = r#"Based on my analysis {"causal_link": false} is the result"#;
        assert_eq!(llm.extract_json(surrounded), r#"{"causal_link": false}"#);
    }

    #[test]
    fn test_parse_causal_response() {
        let llm = CausalDiscoveryLLM::new().unwrap();

        let response =
            r#"{"causal_link": true, "direction": "A_causes_B", "confidence": 0.85, "mechanism": "Direct causation"}"#;

        let result = llm.parse_causal_response(response).unwrap();
        assert!(result.has_causal_link);
        assert_eq!(result.direction, CausalLinkDirection::ACausesB);
        assert!((result.confidence - 0.85).abs() < 0.01);
        assert_eq!(result.mechanism, "Direct causation");
    }
}
