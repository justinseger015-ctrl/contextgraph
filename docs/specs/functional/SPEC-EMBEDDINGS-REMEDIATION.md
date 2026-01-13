# Functional Specification: Embeddings Domain Remediation

## Metadata

| Field | Value |
|-------|-------|
| **Spec ID** | SPEC-EMBED-001 |
| **Title** | Embeddings Domain Remediation - Green Contexts and E12 TokenPruning |
| **Status** | Draft |
| **Version** | 1.0.0 |
| **Created** | 2026-01-12 |
| **Author** | Specification Agent |
| **Related Specs** | SPEC-DREAM-GPU-REMEDIATION (GPU Detection), SPEC-PERFORMANCE-REMEDIATION |
| **Source Issues** | ISS-008, ISS-009 |
| **Constitution Version** | v5.0.0 |
| **Priority** | HIGH |

---

## Overview

### Problem Statement

The Embeddings Domain has two high-priority issues preventing optimal GPU utilization and E12 Late Interaction embedding efficiency:

1. **ISS-008 (HIGH)**: Green Contexts are disabled by default, leaving RTX 5090's compute capability 12.0 features unutilized. The 70%/30% inference/background GPU partition is not auto-enabled.

2. **ISS-009 (HIGH)**: TokenPruning for E12 (Late Interaction) embeddings is not implemented. E12 produces 128D per token, requiring ~50% compression via token pruning to meet storage and performance targets.

### Why This Matters

**Green Contexts (ISS-008)**:
Green Contexts are CUDA's mechanism for partitioning GPU resources between inference workloads and background processing (like dream cycles). Without auto-enabling on compute capability 12.0+:
- RTX 5090's advanced scheduling features are unused
- Dream cycles compete with inference instead of using dedicated partition
- GPU utilization is suboptimal during mixed workloads
- Constitution-mandated 70%/30% partition is not enforced

**TokenPruning for E12 (ISS-009)**:
E12 (Late Interaction) produces dense per-token embeddings at 128D/token. For a typical 512-token document:
- Uncompressed: 512 tokens x 128 dimensions x 4 bytes = 256KB per document
- Target: ~128KB per document (~50% reduction via token pruning)

Without TokenPruning:
- Storage costs are 2x higher than budget
- E12 retrieval (Stage S5 MaxSim) is 2x slower
- Memory pressure prevents scaling to 10M+ nodes

### Constitution Requirements

From `docs2/constitution.yaml`:

```yaml
stack:
  gpu: { target: "RTX 5090", vram: "32GB", compute: "12.0" }

embeddings:
  models:
    E12_LateInteraction: { dim: "128D/tok", type: dense_per_token, use: "Precise match" }

# Implicit requirement: Green Contexts for compute 12.0+
# Implicit requirement: TokenPruning to achieve ~17KB total quantized storage per array
```

### End State

After remediation:
1. Green Contexts auto-enable when `compute_capability >= 12.0` is detected
2. GPU partition enforces 70% inference / 30% background split
3. E12 TokenPruning achieves ~50% compression via attention-based token selection
4. System startup validates GPU compute capability and configures accordingly
5. Runtime detection gracefully degrades on older GPUs

---

## User Stories

### US-EMBED-001: System Administrator - Green Contexts Auto-Enable

```yaml
story_id: US-EMBED-001
priority: must-have
narrative: |
  As a System Administrator deploying on RTX 5090,
  I want Green Contexts to automatically enable when compute capability 12.0+ is detected,
  So that the 70%/30% inference/background partition is enforced without manual configuration.
acceptance_criteria:
  - criterion_id: AC-EMBED-001-01
    given: The MCP server starts on RTX 5090 (compute 12.0)
    when: GPU initialization completes
    then: Green Contexts MUST be enabled automatically
  - criterion_id: AC-EMBED-001-02
    given: The MCP server starts on RTX 4090 (compute 8.9)
    when: GPU initialization completes
    then: Green Contexts MUST remain disabled (not supported)
  - criterion_id: AC-EMBED-001-03
    given: Green Contexts are enabled
    when: I query GPU partition status
    then: I see 70% allocated to inference, 30% to background
  - criterion_id: AC-EMBED-001-04
    given: Green Contexts fail to enable despite supported hardware
    when: Server initialization is in progress
    then: System MUST log warning but NOT fail startup (graceful degradation)
```

### US-EMBED-002: Developer - Runtime GPU Architecture Detection

```yaml
story_id: US-EMBED-002
priority: must-have
narrative: |
  As a Developer,
  I want the system to detect GPU compute capability at runtime,
  So that feature enablement is based on actual hardware, not configuration.
acceptance_criteria:
  - criterion_id: AC-EMBED-002-01
    given: NVIDIA GPU is present
    when: GpuArchitectureDetector::detect() is called
    then: Returns actual compute capability (e.g., 12.0, 8.9, 7.5)
  - criterion_id: AC-EMBED-002-02
    given: No NVIDIA GPU present
    when: GpuArchitectureDetector::detect() is called
    then: Returns Err(GpuDetectionError::NoNvidiaGpu)
  - criterion_id: AC-EMBED-002-03
    given: Multiple GPUs with different compute capabilities
    when: GpuArchitectureDetector::detect() is called
    then: Returns minimum compute capability (most restrictive)
  - criterion_id: AC-EMBED-002-04
    given: Compute capability is detected
    when: I query feature support
    then: Green Contexts report enabled/disabled based on >= 12.0 threshold
```

### US-EMBED-003: System - E12 TokenPruning Implementation

```yaml
story_id: US-EMBED-003
priority: must-have
narrative: |
  As the embedding storage system,
  I want to prune low-importance tokens from E12 embeddings,
  So that storage requirements are reduced by ~50% while preserving retrieval quality.
acceptance_criteria:
  - criterion_id: AC-EMBED-003-01
    given: A document produces 512 E12 token embeddings
    when: TokenPruning is applied
    then: Output contains <= 256 token embeddings (~50% reduction)
  - criterion_id: AC-EMBED-003-02
    given: TokenPruning is applied to E12 embeddings
    when: MaxSim retrieval (Stage S5) is performed
    then: Recall@10 degrades by < 5% compared to unpruned
  - criterion_id: AC-EMBED-003-03
    given: A document has only 100 tokens (below pruning threshold)
    when: TokenPruning is applied
    then: All tokens are retained (no unnecessary pruning)
  - criterion_id: AC-EMBED-003-04
    given: TokenPruning is applied
    when: I compute storage per document
    then: E12 storage is <= 128KB (down from ~256KB unpruned)
```

### US-EMBED-004: System - E12 Compression Target

```yaml
story_id: US-EMBED-004
priority: should-have
narrative: |
  As the embedding storage system,
  I want E12 TokenPruning to achieve consistent ~50% compression,
  So that the 13-embedding array stays within the ~17KB quantized budget.
acceptance_criteria:
  - criterion_id: AC-EMBED-004-01
    given: A batch of 1000 documents with varying lengths
    when: TokenPruning is applied
    then: Average compression ratio is between 45% and 55%
  - criterion_id: AC-EMBED-004-02
    given: TokenPruning is applied with attention-based selection
    when: Token importance scores are computed
    then: Scores correlate with attention weights from E12 model
  - criterion_id: AC-EMBED-004-03
    given: Compression target cannot be met
    when: Document has too few tokens
    then: System logs INFO and retains all tokens
```

---

## Requirements

### REQ-EMBED-001: Green Contexts Auto-Enable on Compute Capability >= 12.0

```yaml
requirement_id: REQ-EMBED-001
story_ref: US-EMBED-001
priority: must
severity: HIGH
constitution_rule: stack.gpu.compute, stack.gpu.target
description: |
  Green Contexts MUST automatically enable when the system detects
  NVIDIA GPU compute capability >= 12.0 (RTX 5090 and later).
rationale: |
  RTX 5090 introduces compute capability 12.0 with enhanced scheduling
  features including Green Contexts for workload partitioning. The
  constitution specifies RTX 5090 as the target GPU, and the 70%/30%
  inference/background split requires Green Contexts for enforcement.
implementation_location:
  - crates/context-graph-cuda/src/context/green_contexts.rs (new file)
  - crates/context-graph-cuda/src/lib.rs (initialization)
pseudocode: |
  pub struct GreenContextsManager {
      enabled: bool,
      inference_partition: f32,  // 0.70
      background_partition: f32, // 0.30
  }

  impl GreenContextsManager {
      pub fn new(compute_cap: ComputeCapability) -> Self {
          let enabled = compute_cap.major >= 12;
          if enabled {
              // Configure CUDA Green Contexts
              cuda_configure_green_contexts(0.70, 0.30)?;
              info!("Green Contexts enabled: 70% inference, 30% background");
          } else {
              info!("Green Contexts disabled: compute capability {} < 12.0",
                    compute_cap);
          }
          Self { enabled, inference_partition: 0.70, background_partition: 0.30 }
      }
  }
fail_fast_behavior: |
  Green Contexts failure is NOT fatal. If enabling fails on supported hardware:
  - Log warning: "Green Contexts failed to enable: {reason}"
  - Continue with default GPU scheduling
  - System operates without partitioning (graceful degradation)
```

### REQ-EMBED-002: Runtime GPU Architecture Detection

```yaml
requirement_id: REQ-EMBED-002
story_ref: US-EMBED-002
priority: must
severity: HIGH
constitution_rule: stack.gpu, ARCH-08
description: |
  The system MUST detect GPU compute capability at runtime using CUDA
  device query APIs. This detection enables feature-based configuration.
rationale: |
  Hardcoding GPU features is fragile. Runtime detection allows:
  - Single binary for all GPU architectures
  - Automatic feature enablement based on hardware
  - Graceful degradation on older GPUs
  - Future-proofing for compute capability 13.0+
implementation_location:
  - crates/context-graph-cuda/src/detection/architecture.rs (new file)
pseudocode: |
  #[derive(Debug, Clone, Copy)]
  pub struct ComputeCapability {
      pub major: u32,
      pub minor: u32,
  }

  impl ComputeCapability {
      pub fn supports_green_contexts(&self) -> bool {
          self.major >= 12
      }

      pub fn as_f32(&self) -> f32 {
          self.major as f32 + (self.minor as f32 / 10.0)
      }
  }

  pub struct GpuArchitectureDetector;

  impl GpuArchitectureDetector {
      pub fn detect() -> Result<ComputeCapability, GpuDetectionError> {
          // Use cuDeviceGetAttribute with CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR
          let device_count = cuda_get_device_count()?;
          if device_count == 0 {
              return Err(GpuDetectionError::NoNvidiaGpu);
          }

          let mut min_cap = ComputeCapability { major: u32::MAX, minor: u32::MAX };
          for device_id in 0..device_count {
              let cap = cuda_get_compute_capability(device_id)?;
              if cap.major < min_cap.major ||
                 (cap.major == min_cap.major && cap.minor < min_cap.minor) {
                  min_cap = cap;
              }
          }
          Ok(min_cap)
      }
  }
acceptance_test: |
  #[test]
  fn test_compute_capability_detection() {
      match GpuArchitectureDetector::detect() {
          Ok(cap) => {
              assert!(cap.major >= 7, "Modern GPU required");
              println!("Detected compute capability: {}.{}", cap.major, cap.minor);
          }
          Err(GpuDetectionError::NoNvidiaGpu) => {
              // OK for non-GPU test environments
          }
          Err(e) => panic!("Unexpected error: {:?}", e),
      }
  }
```

### REQ-EMBED-003: E12 TokenPruning Implementation

```yaml
requirement_id: REQ-EMBED-003
story_ref: US-EMBED-003
priority: must
severity: HIGH
constitution_rule: embeddings.models.E12_LateInteraction
description: |
  E12 Late Interaction embeddings MUST implement TokenPruning to reduce
  the number of per-token embeddings by approximately 50%.
rationale: |
  E12 produces 128D embeddings for EACH token. For a 512-token document:
  - Raw: 512 * 128 * 4 bytes = 256KB
  - Target: ~128KB (50% reduction)

  Without pruning, E12 dominates storage and makes the 13-embedding array
  exceed the ~17KB quantized budget target.
implementation_location:
  - crates/context-graph-embeddings/src/pruning/token_pruner.rs (new file)
  - crates/context-graph-embeddings/src/models/e12_late_interaction.rs
pseudocode: |
  /// Token pruning for E12 Late Interaction embeddings.
  /// Uses attention-based importance scoring to select top-k tokens.
  pub struct E12TokenPruner {
      /// Target retention ratio (0.5 = keep 50% of tokens)
      retention_ratio: f32,
      /// Minimum tokens to always retain
      min_tokens: usize,
      /// Maximum tokens before pruning kicks in
      pruning_threshold: usize,
  }

  impl E12TokenPruner {
      pub fn new() -> Self {
          Self {
              retention_ratio: 0.50,
              min_tokens: 32,
              pruning_threshold: 64,
          }
      }

      /// Prune tokens from E12 embeddings based on attention-based importance.
      ///
      /// # Arguments
      /// * `token_embeddings` - Original 128D embeddings per token
      /// * `attention_weights` - Attention scores from E12 model (optional)
      ///
      /// # Returns
      /// * Pruned token embeddings with ~50% fewer tokens
      pub fn prune(
          &self,
          token_embeddings: &[Vec<f32>],  // [num_tokens][128]
          attention_weights: Option<&[f32]>,
      ) -> Vec<Vec<f32>> {
          let num_tokens = token_embeddings.len();

          // Don't prune if below threshold
          if num_tokens <= self.pruning_threshold {
              return token_embeddings.to_vec();
          }

          // Compute importance scores
          let importance = match attention_weights {
              Some(weights) => self.attention_based_importance(weights),
              None => self.embedding_based_importance(token_embeddings),
          };

          // Select top-k tokens
          let target_count = (num_tokens as f32 * self.retention_ratio)
              .max(self.min_tokens as f32) as usize;

          let mut indices: Vec<usize> = (0..num_tokens).collect();
          indices.sort_by(|&a, &b| importance[b].partial_cmp(&importance[a]).unwrap());
          indices.truncate(target_count);
          indices.sort(); // Restore original order

          indices.iter()
              .map(|&i| token_embeddings[i].clone())
              .collect()
      }

      fn attention_based_importance(&self, weights: &[f32]) -> Vec<f32> {
          // Use attention weights directly as importance scores
          weights.to_vec()
      }

      fn embedding_based_importance(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
          // Compute L2 norm as proxy for importance
          embeddings.iter()
              .map(|emb| emb.iter().map(|x| x * x).sum::<f32>().sqrt())
              .collect()
      }
  }
acceptance_test: |
  #[test]
  fn test_e12_token_pruning_compression() {
      let pruner = E12TokenPruner::new();

      // Simulate 512 tokens, each 128D
      let token_embeddings: Vec<Vec<f32>> = (0..512)
          .map(|_| vec![0.1; 128])
          .collect();

      let pruned = pruner.prune(&token_embeddings, None);

      // Should achieve ~50% reduction
      assert!(pruned.len() <= 256, "Expected <= 256 tokens, got {}", pruned.len());
      assert!(pruned.len() >= 200, "Expected >= 200 tokens, got {}", pruned.len());
  }
```

### REQ-EMBED-004: E12 Compression Target ~50%

```yaml
requirement_id: REQ-EMBED-004
story_ref: US-EMBED-004
priority: should
severity: HIGH
constitution_rule: embeddings.paradigm, perf.memory
description: |
  E12 TokenPruning MUST achieve approximately 50% compression on average,
  measured across documents of varying lengths.
rationale: |
  The ~50% target balances storage reduction with retrieval quality.
  - Too aggressive (>70%): Significant recall degradation
  - Too conservative (<30%): Insufficient storage savings
  - 50% sweet spot: ~5% recall degradation, 2x storage savings
implementation_location:
  - crates/context-graph-embeddings/src/pruning/token_pruner.rs
metrics: |
  | Metric | Target | Validation |
  |--------|--------|------------|
  | Compression Ratio | 45-55% | Batch test on 1000 docs |
  | Recall@10 Degradation | < 5% | MaxSim benchmark |
  | Latency Overhead | < 5ms | Pruning computation |
acceptance_test: |
  #[test]
  fn test_e12_compression_ratio() {
      let pruner = E12TokenPruner::new();
      let mut total_original = 0;
      let mut total_pruned = 0;

      // Test documents of varying lengths
      for doc_len in [64, 128, 256, 512, 1024] {
          let embeddings: Vec<Vec<f32>> = (0..doc_len)
              .map(|_| vec![0.1; 128])
              .collect();

          let pruned = pruner.prune(&embeddings, None);

          total_original += doc_len;
          total_pruned += pruned.len();
      }

      let ratio = total_pruned as f32 / total_original as f32;
      assert!(ratio >= 0.45 && ratio <= 0.55,
          "Compression ratio {} not in target range [0.45, 0.55]", ratio);
  }
```

---

## Edge Cases

### EC-EMBED-001: No NVIDIA GPU Present

```yaml
edge_case_id: EC-EMBED-001
req_ref: REQ-EMBED-001, REQ-EMBED-002
scenario: |
  System runs on hardware without NVIDIA GPU (development laptop, AMD GPU, etc.)
expected_behavior: |
  - GpuArchitectureDetector::detect() returns Err(NoNvidiaGpu)
  - Green Contexts remain disabled
  - System logs warning: "No NVIDIA GPU detected - GPU features disabled"
  - System continues with CPU fallback for embeddings
  - Constitution ARCH-08 violation logged in development mode only
test_case: |
  #[test]
  fn test_no_gpu_graceful_degradation() {
      // Force no-GPU mode
      std::env::set_var("CONTEXT_GRAPH_FORCE_CPU", "1");

      let result = GpuArchitectureDetector::detect();
      assert!(matches!(result, Err(GpuDetectionError::NoNvidiaGpu)));

      let green = GreenContextsManager::new_disabled();
      assert!(!green.enabled);
  }
```

### EC-EMBED-002: Older GPU (Compute Capability < 12.0)

```yaml
edge_case_id: EC-EMBED-002
req_ref: REQ-EMBED-001
scenario: |
  System runs on RTX 4090 (compute 8.9) or older GPU
expected_behavior: |
  - GpuArchitectureDetector detects compute capability 8.9
  - Green Contexts report as unsupported
  - System logs info: "Green Contexts require compute 12.0+, detected 8.9"
  - All other GPU features work normally
  - No error, just feature unavailability
test_case: |
  #[test]
  fn test_older_gpu_no_green_contexts() {
      let cap = ComputeCapability { major: 8, minor: 9 };
      assert!(!cap.supports_green_contexts());

      let green = GreenContextsManager::new(cap);
      assert!(!green.enabled);
  }
```

### EC-EMBED-003: Document with Few Tokens

```yaml
edge_case_id: EC-EMBED-003
req_ref: REQ-EMBED-003
scenario: |
  Document has only 50 tokens, below the pruning threshold (64)
expected_behavior: |
  - TokenPruner detects document below threshold
  - All tokens are retained (no pruning)
  - Log: "Document has {} tokens, below pruning threshold {}"
  - Original embeddings returned unchanged
test_case: |
  #[test]
  fn test_small_document_no_pruning() {
      let pruner = E12TokenPruner::new();
      let embeddings: Vec<Vec<f32>> = (0..50)
          .map(|_| vec![0.1; 128])
          .collect();

      let pruned = pruner.prune(&embeddings, None);

      // Should retain all tokens
      assert_eq!(pruned.len(), 50);
  }
```

### EC-EMBED-004: Very Long Document

```yaml
edge_case_id: EC-EMBED-004
req_ref: REQ-EMBED-003, REQ-EMBED-004
scenario: |
  Document has 4096+ tokens (maximum context length)
expected_behavior: |
  - TokenPruner applies 50% retention
  - Output: ~2048 tokens
  - Compression ratio maintained
  - No memory overflow during processing
test_case: |
  #[test]
  fn test_long_document_pruning() {
      let pruner = E12TokenPruner::new();
      let embeddings: Vec<Vec<f32>> = (0..4096)
          .map(|_| vec![0.1; 128])
          .collect();

      let pruned = pruner.prune(&embeddings, None);

      // Should achieve ~50% reduction
      assert!(pruned.len() <= 2048);
      assert!(pruned.len() >= 1800);
  }
```

### EC-EMBED-005: Multi-GPU with Different Compute Capabilities

```yaml
edge_case_id: EC-EMBED-005
req_ref: REQ-EMBED-002
scenario: |
  System has RTX 5090 (12.0) and RTX 4090 (8.9) GPUs
expected_behavior: |
  - Detection returns MINIMUM capability (8.9)
  - Green Contexts disabled (most restrictive)
  - Log: "Multiple GPUs detected with varying compute capabilities"
  - Log: "Using minimum compute 8.9 for feature decisions"
  - Prevents enabling features unavailable on any GPU
test_case: |
  #[test]
  fn test_multi_gpu_minimum_capability() {
      // Mock detection with multiple GPUs
      let caps = vec![
          ComputeCapability { major: 12, minor: 0 },
          ComputeCapability { major: 8, minor: 9 },
      ];

      let min = caps.iter()
          .min_by(|a, b| {
              (a.major, a.minor).cmp(&(b.major, b.minor))
          })
          .unwrap();

      assert_eq!(min.major, 8);
      assert_eq!(min.minor, 9);
      assert!(!min.supports_green_contexts());
  }
```

### EC-EMBED-006: Green Contexts Enable Failure

```yaml
edge_case_id: EC-EMBED-006
req_ref: REQ-EMBED-001
scenario: |
  RTX 5090 detected but Green Contexts CUDA API fails
expected_behavior: |
  - Compute capability correctly detected as 12.0
  - Green Contexts enable attempt fails
  - Log warning: "Green Contexts failed to enable: {cuda_error}"
  - System continues without partitioning
  - No fatal error, graceful degradation
test_case: |
  #[test]
  fn test_green_contexts_failure_graceful() {
      let cap = ComputeCapability { major: 12, minor: 0 };
      assert!(cap.supports_green_contexts());

      // Simulate CUDA failure during enable
      let result = GreenContextsManager::try_enable(cap);
      match result {
          Ok(manager) => assert!(manager.enabled),
          Err(e) => {
              // Graceful degradation
              let manager = GreenContextsManager::new_disabled();
              assert!(!manager.enabled);
          }
      }
  }
```

---

## Error States

### ERR-EMBED-001: GPU Detection Failed

```yaml
error_id: ERR-EMBED-001
http_code: N/A (startup)
condition: CUDA driver not available or failed to query devices
message: |
  GPU detection failed: {reason}
  System will continue with CPU mode.
recovery: |
  1. Install NVIDIA CUDA drivers
  2. Verify nvidia-smi shows GPU
  3. Restart the system
log_level: WARN
```

### ERR-EMBED-002: No NVIDIA GPU Found

```yaml
error_id: ERR-EMBED-002
http_code: N/A (startup)
condition: CUDA reports 0 devices
message: |
  No NVIDIA GPU detected. GPU features disabled.
  Constitution ARCH-08 requires GPU for production.
recovery: |
  Development: Acceptable, use CPU fallback
  Production: Add NVIDIA GPU or use GPU-enabled instance
log_level: WARN (dev), ERROR (prod)
```

### ERR-EMBED-003: Green Contexts Enable Failed

```yaml
error_id: ERR-EMBED-003
http_code: N/A (startup)
condition: Green Contexts CUDA API returns error on supported hardware
message: |
  Green Contexts failed to enable on compute {version}: {cuda_error}
  Continuing without GPU partitioning.
recovery: |
  1. Update CUDA drivers to latest version
  2. Verify RTX 5090 firmware is current
  3. Check for CUDA context conflicts
log_level: WARN
```

### ERR-EMBED-004: TokenPruning Allocation Failed

```yaml
error_id: ERR-EMBED-004
http_code: N/A (runtime)
condition: Memory allocation fails during token pruning
message: |
  E12 TokenPruning failed: insufficient memory for {num_tokens} tokens
  Document embedding stored without pruning.
recovery: |
  System continues with unpruned embeddings for this document.
  Future documents will retry pruning.
log_level: WARN
```

---

## Test Plan

### Unit Tests

#### TC-EMBED-001: GPU Compute Capability Detection

```yaml
test_case_id: TC-EMBED-001
type: unit
req_ref: REQ-EMBED-002
description: Verify compute capability detection returns valid values
inputs: System with NVIDIA GPU
expected: ComputeCapability with major >= 7
test_code: |
  #[test]
  fn test_compute_capability_detection() {
      match GpuArchitectureDetector::detect() {
          Ok(cap) => {
              assert!(cap.major >= 7, "Expected modern GPU, got {}.{}", cap.major, cap.minor);
              let f = cap.as_f32();
              assert!(f >= 7.0 && f <= 15.0, "Invalid capability {}", f);
          }
          Err(GpuDetectionError::NoNvidiaGpu) => {
              // Skip on non-GPU systems
          }
          Err(e) => panic!("Unexpected error: {:?}", e),
      }
  }
```

#### TC-EMBED-002: Green Contexts Feature Detection

```yaml
test_case_id: TC-EMBED-002
type: unit
req_ref: REQ-EMBED-001
description: Verify Green Contexts feature detection based on compute capability
inputs: Various compute capabilities
expected: Enabled only for >= 12.0
test_code: |
  #[test]
  fn test_green_contexts_feature_detection() {
      // RTX 5090 (12.0) - supported
      let cap_12 = ComputeCapability { major: 12, minor: 0 };
      assert!(cap_12.supports_green_contexts());

      // RTX 4090 (8.9) - not supported
      let cap_89 = ComputeCapability { major: 8, minor: 9 };
      assert!(!cap_89.supports_green_contexts());

      // Future GPU (13.0) - supported
      let cap_13 = ComputeCapability { major: 13, minor: 0 };
      assert!(cap_13.supports_green_contexts());

      // RTX 3090 (8.6) - not supported
      let cap_86 = ComputeCapability { major: 8, minor: 6 };
      assert!(!cap_86.supports_green_contexts());
  }
```

#### TC-EMBED-003: E12 TokenPruning Basic

```yaml
test_case_id: TC-EMBED-003
type: unit
req_ref: REQ-EMBED-003
description: Verify basic token pruning achieves target compression
inputs: 512 token embeddings
expected: ~256 tokens retained
test_code: |
  #[test]
  fn test_e12_token_pruning_basic() {
      let pruner = E12TokenPruner::new();

      let embeddings: Vec<Vec<f32>> = (0..512)
          .map(|i| {
              // Varying importance via embedding magnitude
              let scale = 0.1 + (i as f32 / 512.0) * 0.9;
              vec![scale; 128]
          })
          .collect();

      let pruned = pruner.prune(&embeddings, None);

      // ~50% retention
      assert!(pruned.len() >= 230 && pruned.len() <= 280,
          "Expected ~256 tokens, got {}", pruned.len());
  }
```

#### TC-EMBED-004: E12 TokenPruning with Attention Weights

```yaml
test_case_id: TC-EMBED-004
type: unit
req_ref: REQ-EMBED-003
description: Verify attention-based pruning preserves high-attention tokens
inputs: Embeddings with attention weights
expected: High-attention tokens retained
test_code: |
  #[test]
  fn test_e12_token_pruning_attention_based() {
      let pruner = E12TokenPruner::new();

      let embeddings: Vec<Vec<f32>> = (0..512)
          .map(|_| vec![0.1; 128])
          .collect();

      // High attention for tokens 0-99, low for rest
      let mut attention: Vec<f32> = vec![0.01; 512];
      for i in 0..100 {
          attention[i] = 0.9;
      }

      let pruned = pruner.prune(&embeddings, Some(&attention));

      // High-attention tokens should be retained
      // Pruned count should include most of the first 100
      assert!(pruned.len() >= 100 && pruned.len() <= 280);
  }
```

#### TC-EMBED-005: E12 TokenPruning Below Threshold

```yaml
test_case_id: TC-EMBED-005
type: unit
req_ref: REQ-EMBED-003
description: Verify no pruning for documents below threshold
inputs: 50 token embeddings (below 64 threshold)
expected: All 50 tokens retained
test_code: |
  #[test]
  fn test_e12_token_pruning_below_threshold() {
      let pruner = E12TokenPruner::new();

      let embeddings: Vec<Vec<f32>> = (0..50)
          .map(|_| vec![0.1; 128])
          .collect();

      let pruned = pruner.prune(&embeddings, None);

      // No pruning below threshold
      assert_eq!(pruned.len(), 50);
  }
```

#### TC-EMBED-006: Compression Ratio Validation

```yaml
test_case_id: TC-EMBED-006
type: unit
req_ref: REQ-EMBED-004
description: Verify compression ratio is consistently 45-55%
inputs: Batch of documents with varying lengths
expected: Average ratio in [0.45, 0.55]
test_code: |
  #[test]
  fn test_compression_ratio_batch() {
      let pruner = E12TokenPruner::new();
      let mut ratios = Vec::new();

      for doc_len in [128, 256, 384, 512, 768, 1024] {
          let embeddings: Vec<Vec<f32>> = (0..doc_len)
              .map(|_| vec![0.1; 128])
              .collect();

          let pruned = pruner.prune(&embeddings, None);
          let ratio = pruned.len() as f32 / doc_len as f32;
          ratios.push(ratio);
      }

      let avg_ratio: f32 = ratios.iter().sum::<f32>() / ratios.len() as f32;
      assert!(avg_ratio >= 0.45 && avg_ratio <= 0.55,
          "Average compression {} not in target range", avg_ratio);
  }
```

### Integration Tests

#### TC-EMBED-INT-001: Full GPU Initialization Flow

```yaml
test_case_id: TC-EMBED-INT-001
type: integration
req_ref: REQ-EMBED-001, REQ-EMBED-002
description: Test complete GPU detection and Green Contexts initialization
preconditions: RTX 5090 or test environment
steps: |
  1. Initialize GpuArchitectureDetector
  2. Detect compute capability
  3. Create GreenContextsManager based on capability
  4. Verify partition ratios if enabled
expected: |
  - On 12.0+: Green Contexts enabled, 70/30 partition
  - On <12.0: Green Contexts disabled, no partition
  - On no GPU: Graceful degradation
test_code: |
  #[tokio::test]
  async fn test_full_gpu_initialization() {
      let result = GpuArchitectureDetector::detect();

      match result {
          Ok(cap) => {
              let green = GreenContextsManager::new(cap);
              if cap.supports_green_contexts() {
                  assert!(green.enabled);
                  assert_eq!(green.inference_partition, 0.70);
                  assert_eq!(green.background_partition, 0.30);
              } else {
                  assert!(!green.enabled);
              }
          }
          Err(GpuDetectionError::NoNvidiaGpu) => {
              let green = GreenContextsManager::new_disabled();
              assert!(!green.enabled);
          }
          Err(e) => panic!("Unexpected error: {:?}", e),
      }
  }
```

#### TC-EMBED-INT-002: E12 Pipeline with TokenPruning

```yaml
test_case_id: TC-EMBED-INT-002
type: integration
req_ref: REQ-EMBED-003, REQ-EMBED-004
description: Test E12 embedding pipeline with TokenPruning integrated
preconditions: E12 model loaded
steps: |
  1. Encode document through E12 model
  2. Apply TokenPruning to output
  3. Verify compression achieved
  4. Verify MaxSim still works on pruned embeddings
test_code: |
  #[tokio::test]
  async fn test_e12_pipeline_with_pruning() {
      let e12 = E12LateInteraction::load().await?;
      let pruner = E12TokenPruner::new();

      let text = "A moderately long document for testing purposes...";
      let embeddings = e12.encode(text).await?;

      let original_count = embeddings.len();
      let pruned = pruner.prune(&embeddings, e12.last_attention_weights());

      // Verify compression
      if original_count > 64 {
          let ratio = pruned.len() as f32 / original_count as f32;
          assert!(ratio >= 0.45 && ratio <= 0.55);
      }

      // Verify MaxSim still works
      let query_emb = e12.encode("testing").await?;
      let score = maxsim(&query_emb, &pruned);
      assert!(score >= 0.0 && score <= 1.0);
  }
```

### Constitution Compliance Tests

#### TC-CONST-EMBED-001: RTX 5090 Target

```yaml
test_case_id: TC-CONST-EMBED-001
type: constitution
related: stack.gpu.target
description: Verify system targets RTX 5090 compute 12.0
expected: |
  - Compute 12.0 features available when hardware present
  - Green Contexts auto-enable on RTX 5090
test_code: |
  #[test]
  fn test_constitution_rtx_5090_target() {
      let target_cap = ComputeCapability { major: 12, minor: 0 };
      assert!(target_cap.supports_green_contexts(),
          "Constitution stack.gpu.target: RTX 5090 must support Green Contexts");
  }
```

#### TC-CONST-EMBED-002: E12 128D/tok Dimension

```yaml
test_case_id: TC-CONST-EMBED-002
type: constitution
related: embeddings.models.E12_LateInteraction
description: Verify E12 produces 128D per-token embeddings
expected: Each token embedding is 128-dimensional
test_code: |
  #[test]
  fn test_constitution_e12_dimension() {
      let pruner = E12TokenPruner::new();

      // Simulate E12 output with correct dimensions
      let embeddings: Vec<Vec<f32>> = (0..100)
          .map(|_| vec![0.1; 128])  // 128D per token
          .collect();

      let pruned = pruner.prune(&embeddings, None);

      // All pruned embeddings should retain 128D
      for emb in &pruned {
          assert_eq!(emb.len(), 128,
              "Constitution E12: 128D/tok required");
      }
  }
```

---

## Dependency Graph

```
REQ-EMBED-002 (GPU Architecture Detection)
    |
    +-- REQ-EMBED-001 (Green Contexts Auto-Enable)
    |       |
    |       +-- 70%/30% Partition Enforcement
    |       |
    |       +-- Dream Background Processing
    |
    +-- Other GPU Features (future)

REQ-EMBED-003 (E12 TokenPruning)
    |
    +-- REQ-EMBED-004 (~50% Compression Target)
            |
            +-- E12 Storage Optimization
            |
            +-- MaxSim Retrieval Performance
```

### Implementation Order

1. **REQ-EMBED-002**: GPU Architecture Detection (foundation)
2. **REQ-EMBED-001**: Green Contexts Auto-Enable (depends on #1)
3. **REQ-EMBED-003**: E12 TokenPruning Implementation
4. **REQ-EMBED-004**: Compression Target Validation

### External Dependencies

| Dependency | Source | Required |
|------------|--------|----------|
| CUDA Toolkit 13.1+ | NVIDIA | Yes (for Green Contexts) |
| `cudarc` crate | crates.io | Yes |
| RTX 5090 hardware | Hardware | No (graceful degradation) |
| E12 model weights | Local | Yes (for TokenPruning) |

---

## Files to Modify/Create

| File | Type | Changes |
|------|------|---------|
| `crates/context-graph-cuda/src/detection/architecture.rs` | New | GPU architecture detection |
| `crates/context-graph-cuda/src/context/green_contexts.rs` | New | Green Contexts manager |
| `crates/context-graph-cuda/src/lib.rs` | Modify | Add detection and context modules |
| `crates/context-graph-embeddings/src/pruning/token_pruner.rs` | New | E12 TokenPruning |
| `crates/context-graph-embeddings/src/pruning/mod.rs` | New | Pruning module |
| `crates/context-graph-embeddings/src/models/e12_late_interaction.rs` | Modify | Integrate TokenPruning |
| `crates/context-graph-embeddings/Cargo.toml` | Modify | Add dependencies |

---

## Estimated Effort

| Task | Effort | Notes |
|------|--------|-------|
| GPU Architecture Detection | 2-3h | CUDA device queries |
| Green Contexts Manager | 3-4h | CUDA 13.1 API integration |
| E12 TokenPruner struct | 2-3h | Core pruning logic |
| Attention-based importance | 1-2h | Integration with E12 model |
| Unit tests | 3-4h | All test cases |
| Integration tests | 2-3h | End-to-end validation |
| Documentation | 1h | Update docs |
| **Total** | **14-20h** | |

---

## Success Criteria

1. Green Contexts auto-enable on RTX 5090 (compute 12.0+)
2. GPU architecture detection works on all supported hardware
3. E12 TokenPruning achieves 45-55% compression ratio
4. MaxSim recall@10 degrades by < 5% with pruning
5. All constitution compliance tests pass
6. Graceful degradation on unsupported hardware
7. No startup failures due to GPU feature issues

---

## Appendix: Constitution References

### stack.gpu (Target Hardware)

```yaml
stack:
  gpu: { target: "RTX 5090", vram: "32GB", compute: "12.0" }
```

### embeddings.models.E12_LateInteraction

```yaml
embeddings:
  models:
    E12_LateInteraction: { dim: "128D/tok", type: dense_per_token, use: "Precise match" }
```

### Green Contexts Partition (Implicit)

```yaml
# Derived from dream.constraints.gpu and perf requirements
# Green Contexts partition: 70% inference / 30% background
# Background partition used for dream cycles
```

### perf.memory (Storage Budget)

```yaml
perf:
  memory: { gpu: "<24GB", graph_cap: ">10M nodes" }
  # ~17KB quantized per embedding array implies E12 must be compressed
```

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-12 | Specification Agent | Initial specification |
