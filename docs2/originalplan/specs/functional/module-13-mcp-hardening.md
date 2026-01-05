# Module 13: MCP Hardening - Functional Specification

**Module ID**: SPEC-MCPH-013
**Version**: 1.0.0
**Status**: Draft
**Phase**: 12
**Duration**: 4 weeks
**Dependencies**: Module 1 (Ghost System), Module 2 (Core Infrastructure), Module 6 (Formal Verification), Module 8 (Inference Engine), Module 11 (Immune System), Module 12 (Active Inference), Module 12.5 (Steering Subsystem)
**Last Updated**: 2025-12-31

---

## 1. Executive Summary

The MCP Hardening module fortifies the MCP server for production deployment with comprehensive security measures, rate limiting, authentication, audit logging, and resource management. This module transforms the development-ready MCP interface into a production-grade, security-hardened API layer that can withstand adversarial inputs, denial-of-service attacks, and abuse patterns while maintaining the performance characteristics required by the Ultimate Context Graph system.

### 1.1 Core Objectives

- Implement defense-in-depth security architecture with input validation, sanitization, and output encoding
- Deploy token bucket rate limiting with per-session and global limits
- Provide session-based authentication with permission levels and expiry management
- Enable tamper-evident audit logging with structured records and retention policies
- Enforce resource quotas for memory, CPU time, and storage per session
- Implement graceful degradation under load with circuit breaker patterns

### 1.2 Key Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Validation Overhead | <1ms per request | P99 latency for input validation |
| Rate Limit Check | <100us | P99 latency for rate limit evaluation |
| Auth Verification | <500us | P99 latency for session token verification |
| Audit Log Write | <200us async | P99 latency for non-blocking audit write |
| Attack Detection Rate | >99% | Known attack pattern detection |
| False Positive Rate | <0.1% | Legitimate requests incorrectly blocked |
| Error Information Leakage | 0 | Security audit finding count |
| Uptime Under Attack | >99.9% | Availability during simulated DoS |

### 1.3 Security Posture Goals

| Security Domain | Target State |
|----------------|--------------|
| Input Validation | All inputs validated against strict schemas before processing |
| Authentication | Session-based tokens with 24hr expiry, refresh capability |
| Authorization | Role-based permission levels (read-only, standard, admin) |
| Rate Limiting | 1000 req/min global, 100 req/min per session default |
| Audit Trail | All operations logged with tamper-evident checksums |
| Error Handling | Sanitized errors with no stack traces or internal details |
| DoS Protection | Circuit breakers, request queuing, graceful degradation |

---

## 2. Theoretical Background

### 2.1 Defense in Depth

The MCP Hardening module implements multiple security layers that each provide independent protection:

```
Request Flow:
  [External Input]
        |
        v
  [TLS Termination] -----> Layer 0: Transport Security
        |
        v
  [Rate Limiter] --------> Layer 1: Volume Protection
        |
        v
  [Authentication] ------> Layer 2: Identity Verification
        |
        v
  [Input Validation] ----> Layer 3: Content Security
        |
        v
  [Authorization] -------> Layer 4: Permission Enforcement
        |
        v
  [Request Handler] -----> Layer 5: Business Logic
        |
        v
  [Response Sanitizer] --> Layer 6: Output Security
        |
        v
  [Audit Logger] --------> Layer 7: Accountability
        |
        v
  [Sanitized Response]
```

### 2.2 Token Bucket Algorithm

Rate limiting uses the token bucket algorithm for smooth traffic shaping:

```
Token Bucket:
- capacity: Maximum burst size
- tokens: Current available tokens
- refill_rate: Tokens added per second
- last_refill: Timestamp of last refill

acquire(n):
  refill_tokens()
  if tokens >= n:
    tokens -= n
    return Allow
  else:
    return Deny

refill_tokens():
  elapsed = now() - last_refill
  tokens = min(capacity, tokens + elapsed * refill_rate)
  last_refill = now()
```

### 2.3 Security Requirement Traceability

The module addresses security requirements from the project constitution:

| Constitution ID | Requirement | Implementation |
|----------------|-------------|----------------|
| SEC-01 | Dependency audit with cargo-audit | CI pipeline integration |
| SEC-02 | Input sanitization for all MCP endpoints | InputValidator component |
| SEC-03 | Embedding poisoning detection | Module 11 integration |
| SEC-04 | Memory isolation between sessions | SessionManager isolation |
| SEC-05 | Semantic cancer detection | Module 11 integration |
| (Rate Limits) | 1000 req/min default, 100 req/min per session | RateLimiter component |

---

## 3. Functional Requirements

### 3.1 Input Validation and Sanitization

#### REQ-MCPH-001: InputValidator Struct Definition

**Priority**: Critical
**Description**: The system SHALL implement an InputValidator struct that validates and sanitizes all incoming MCP requests before processing.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use regex::Regex;

/// Input validator implementing defense-in-depth input security.
///
/// Validates all incoming requests against strict schemas, detects injection
/// attempts, enforces size limits, and sanitizes content before processing.
///
/// `Constraint: Validation_Overhead < 1ms per request`
pub struct InputValidator {
    /// Schema definitions for each MCP tool
    pub schemas: HashMap<String, ToolSchema>,

    /// Compiled regex patterns for injection detection
    pub injection_patterns: Vec<CompiledPattern>,

    /// Size limits configuration
    pub size_limits: SizeLimits,

    /// Content sanitizer
    pub sanitizer: ContentSanitizer,

    /// Validation metrics
    pub metrics: InputValidationMetrics,

    /// Configuration
    pub config: InputValidatorConfig,
}

/// Configuration for input validation
#[derive(Clone, Debug)]
pub struct InputValidatorConfig {
    /// Maximum request size in bytes
    pub max_request_size: usize,  // Default: 1MB (1_048_576)

    /// Maximum string field length
    pub max_string_length: usize,  // Default: 65536 (64KB)

    /// Maximum array elements
    pub max_array_elements: usize,  // Default: 1000

    /// Maximum nesting depth for JSON
    pub max_nesting_depth: usize,  // Default: 10

    /// Enable strict mode (reject unknown fields)
    pub strict_mode: bool,  // Default: true

    /// Validation timeout
    pub validation_timeout: Duration,  // Default: 1ms
}

impl Default for InputValidatorConfig {
    fn default() -> Self {
        Self {
            max_request_size: 1_048_576,
            max_string_length: 65536,
            max_array_elements: 1000,
            max_nesting_depth: 10,
            strict_mode: true,
            validation_timeout: Duration::from_millis(1),
        }
    }
}

/// Schema definition for MCP tool validation
#[derive(Clone, Debug)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,

    /// Required parameters
    pub required_params: Vec<ParamSchema>,

    /// Optional parameters
    pub optional_params: Vec<ParamSchema>,

    /// Custom validation rules
    pub custom_rules: Vec<ValidationRule>,
}

/// Parameter schema for validation
#[derive(Clone, Debug)]
pub struct ParamSchema {
    /// Parameter name
    pub name: String,

    /// Expected type
    pub param_type: ParamType,

    /// Minimum value (for numbers) or length (for strings)
    pub min: Option<i64>,

    /// Maximum value (for numbers) or length (for strings)
    pub max: Option<i64>,

    /// Regex pattern for string validation
    pub pattern: Option<String>,

    /// Allowed enum values
    pub allowed_values: Option<Vec<String>>,
}

/// Parameter types for schema validation
#[derive(Clone, Debug, PartialEq)]
pub enum ParamType {
    String,
    Integer,
    Float,
    Boolean,
    Uuid,
    Array(Box<ParamType>),
    Object,
}

/// Custom validation rule
#[derive(Clone, Debug)]
pub enum ValidationRule {
    /// Require at least one of the specified fields
    RequireOneOf(Vec<String>),
    /// Fields are mutually exclusive
    MutuallyExclusive(Vec<String>),
    /// Conditional requirement
    ConditionalRequired { if_field: String, then_required: Vec<String> },
    /// Custom validator function name
    CustomValidator(String),
}

/// Compiled injection detection pattern
#[derive(Clone)]
pub struct CompiledPattern {
    /// Pattern name for logging
    pub name: String,

    /// Compiled regex
    pub regex: Regex,

    /// Severity level
    pub severity: ThreatSeverity,

    /// Action to take on match
    pub action: ThreatAction,
}

/// Threat severity levels
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThreatSeverity {
    /// Low severity - log only
    Low,
    /// Medium severity - log and flag
    Medium,
    /// High severity - block request
    High,
    /// Critical severity - block and alert
    Critical,
}

/// Actions for threat response
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThreatAction {
    /// Log the attempt only
    LogOnly,
    /// Sanitize the content
    Sanitize,
    /// Reject the request
    Reject,
    /// Reject and quarantine session
    QuarantineSession,
}

/// Size limits for various input types
#[derive(Clone, Debug)]
pub struct SizeLimits {
    /// Maximum content field size (for store_memory, inject_context query)
    pub max_content_size: usize,  // Default: 65536

    /// Maximum query string size
    pub max_query_size: usize,  // Default: 4096

    /// Maximum metadata object size
    pub max_metadata_size: usize,  // Default: 8192

    /// Maximum number of node IDs in batch operations
    pub max_batch_size: usize,  // Default: 100

    /// Maximum embedding dimension
    pub max_embedding_dim: usize,  // Default: 4096
}

impl Default for SizeLimits {
    fn default() -> Self {
        Self {
            max_content_size: 65536,
            max_query_size: 4096,
            max_metadata_size: 8192,
            max_batch_size: 100,
            max_embedding_dim: 4096,
        }
    }
}

/// Content sanitizer for removing dangerous content
#[derive(Clone)]
pub struct ContentSanitizer {
    /// HTML/script tag patterns to remove
    pub html_patterns: Vec<Regex>,

    /// SQL injection patterns to neutralize
    pub sql_patterns: Vec<Regex>,

    /// Path traversal patterns to block
    pub path_patterns: Vec<Regex>,

    /// Control character removal enabled
    pub strip_control_chars: bool,

    /// Unicode normalization form
    pub unicode_normalization: UnicodeNormalization,
}

/// Unicode normalization forms
#[derive(Clone, Copy, Debug)]
pub enum UnicodeNormalization {
    /// NFC (Canonical Decomposition, followed by Canonical Composition)
    Nfc,
    /// NFD (Canonical Decomposition)
    Nfd,
    /// NFKC (Compatibility Decomposition, followed by Canonical Composition)
    Nfkc,
    /// No normalization
    None,
}

/// Metrics for input validation
#[derive(Clone, Default)]
pub struct InputValidationMetrics {
    /// Total requests validated
    pub total_validated: u64,

    /// Requests rejected
    pub total_rejected: u64,

    /// Injection attempts detected
    pub injection_attempts: u64,

    /// Size limit violations
    pub size_violations: u64,

    /// Schema validation failures
    pub schema_failures: u64,

    /// Average validation latency (microseconds)
    pub avg_validation_latency_us: f64,

    /// P99 validation latency (microseconds)
    pub p99_validation_latency_us: f64,
}

impl InputValidator {
    /// Validate an incoming MCP request
    ///
    /// Returns validated and sanitized request or validation error.
    ///
    /// `Constraint: Validation_Latency < 1ms`
    pub fn validate_request(
        &mut self,
        tool_name: &str,
        params: &serde_json::Value,
    ) -> Result<ValidatedRequest, ValidationError> {
        let start = std::time::Instant::now();

        // Step 1: Check request size
        let request_size = params.to_string().len();
        if request_size > self.config.max_request_size {
            self.metrics.size_violations += 1;
            return Err(ValidationError::RequestTooLarge {
                size: request_size,
                max: self.config.max_request_size,
            });
        }

        // Step 2: Get schema for tool
        let schema = self.schemas.get(tool_name)
            .ok_or_else(|| ValidationError::UnknownTool(tool_name.to_string()))?;

        // Step 3: Validate against schema
        self.validate_schema(params, schema)?;

        // Step 4: Check for injection patterns
        self.check_injections(params)?;

        // Step 5: Sanitize content
        let sanitized = self.sanitizer.sanitize(params)?;

        // Step 6: Validate nesting depth
        self.check_nesting_depth(&sanitized, 0)?;

        // Update metrics
        let elapsed = start.elapsed().as_micros() as f64;
        self.metrics.total_validated += 1;
        self.metrics.avg_validation_latency_us =
            (self.metrics.avg_validation_latency_us * 0.99) + (elapsed * 0.01);

        Ok(ValidatedRequest {
            tool_name: tool_name.to_string(),
            params: sanitized,
            validation_timestamp: std::time::Instant::now(),
        })
    }

    /// Validate parameters against schema
    fn validate_schema(
        &self,
        params: &serde_json::Value,
        schema: &ToolSchema,
    ) -> Result<(), ValidationError> {
        let params_obj = params.as_object()
            .ok_or(ValidationError::InvalidType {
                field: "params".to_string(),
                expected: "object".to_string(),
            })?;

        // Check required parameters
        for required in &schema.required_params {
            let value = params_obj.get(&required.name)
                .ok_or_else(|| ValidationError::MissingRequired(required.name.clone()))?;

            self.validate_param(value, required)?;
        }

        // Check optional parameters if present
        for optional in &schema.optional_params {
            if let Some(value) = params_obj.get(&optional.name) {
                self.validate_param(value, optional)?;
            }
        }

        // Apply custom rules
        for rule in &schema.custom_rules {
            self.apply_custom_rule(params_obj, rule)?;
        }

        // Strict mode: reject unknown fields
        if self.config.strict_mode {
            let known_fields: std::collections::HashSet<_> = schema.required_params.iter()
                .chain(schema.optional_params.iter())
                .map(|p| &p.name)
                .collect();

            for key in params_obj.keys() {
                if !known_fields.contains(key) {
                    return Err(ValidationError::UnknownField(key.clone()));
                }
            }
        }

        Ok(())
    }

    /// Validate a single parameter
    fn validate_param(
        &self,
        value: &serde_json::Value,
        schema: &ParamSchema,
    ) -> Result<(), ValidationError> {
        match &schema.param_type {
            ParamType::String => {
                let s = value.as_str()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: schema.name.clone(),
                        expected: "string".to_string(),
                    })?;

                // Check length
                if let Some(min) = schema.min {
                    if (s.len() as i64) < min {
                        return Err(ValidationError::StringTooShort {
                            field: schema.name.clone(),
                            min: min as usize,
                        });
                    }
                }
                if let Some(max) = schema.max {
                    if (s.len() as i64) > max {
                        return Err(ValidationError::StringTooLong {
                            field: schema.name.clone(),
                            max: max as usize,
                        });
                    }
                }

                // Check pattern
                if let Some(ref pattern) = schema.pattern {
                    let regex = Regex::new(pattern)
                        .map_err(|_| ValidationError::InvalidPattern(pattern.clone()))?;
                    if !regex.is_match(s) {
                        return Err(ValidationError::PatternMismatch {
                            field: schema.name.clone(),
                            pattern: pattern.clone(),
                        });
                    }
                }

                // Check allowed values
                if let Some(ref allowed) = schema.allowed_values {
                    if !allowed.contains(&s.to_string()) {
                        return Err(ValidationError::InvalidEnumValue {
                            field: schema.name.clone(),
                            value: s.to_string(),
                            allowed: allowed.clone(),
                        });
                    }
                }
            }
            ParamType::Integer => {
                let n = value.as_i64()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: schema.name.clone(),
                        expected: "integer".to_string(),
                    })?;

                if let Some(min) = schema.min {
                    if n < min {
                        return Err(ValidationError::NumberTooSmall {
                            field: schema.name.clone(),
                            min,
                        });
                    }
                }
                if let Some(max) = schema.max {
                    if n > max {
                        return Err(ValidationError::NumberTooLarge {
                            field: schema.name.clone(),
                            max,
                        });
                    }
                }
            }
            ParamType::Float => {
                let _n = value.as_f64()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: schema.name.clone(),
                        expected: "float".to_string(),
                    })?;
                // Float range validation similar to integer
            }
            ParamType::Boolean => {
                value.as_bool()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: schema.name.clone(),
                        expected: "boolean".to_string(),
                    })?;
            }
            ParamType::Uuid => {
                let s = value.as_str()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: schema.name.clone(),
                        expected: "uuid string".to_string(),
                    })?;
                Uuid::parse_str(s)
                    .map_err(|_| ValidationError::InvalidUuid {
                        field: schema.name.clone(),
                        value: s.to_string(),
                    })?;
            }
            ParamType::Array(inner_type) => {
                let arr = value.as_array()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: schema.name.clone(),
                        expected: "array".to_string(),
                    })?;

                if arr.len() > self.config.max_array_elements {
                    return Err(ValidationError::ArrayTooLarge {
                        field: schema.name.clone(),
                        size: arr.len(),
                        max: self.config.max_array_elements,
                    });
                }

                // Validate each element
                let inner_schema = ParamSchema {
                    name: format!("{}[]", schema.name),
                    param_type: (**inner_type).clone(),
                    min: None,
                    max: None,
                    pattern: None,
                    allowed_values: None,
                };
                for elem in arr {
                    self.validate_param(elem, &inner_schema)?;
                }
            }
            ParamType::Object => {
                value.as_object()
                    .ok_or_else(|| ValidationError::InvalidType {
                        field: schema.name.clone(),
                        expected: "object".to_string(),
                    })?;
            }
        }

        Ok(())
    }

    /// Check for injection patterns
    fn check_injections(&mut self, params: &serde_json::Value) -> Result<(), ValidationError> {
        let content = params.to_string();

        for pattern in &self.injection_patterns {
            if pattern.regex.is_match(&content) {
                self.metrics.injection_attempts += 1;

                match pattern.action {
                    ThreatAction::LogOnly => {
                        // Log but continue
                        tracing::warn!(
                            pattern = %pattern.name,
                            severity = ?pattern.severity,
                            "Injection pattern detected (log only)"
                        );
                    }
                    ThreatAction::Sanitize => {
                        // Content will be sanitized in next step
                        tracing::warn!(
                            pattern = %pattern.name,
                            severity = ?pattern.severity,
                            "Injection pattern detected (sanitizing)"
                        );
                    }
                    ThreatAction::Reject => {
                        return Err(ValidationError::InjectionDetected {
                            pattern: pattern.name.clone(),
                            severity: pattern.severity,
                        });
                    }
                    ThreatAction::QuarantineSession => {
                        return Err(ValidationError::InjectionDetected {
                            pattern: pattern.name.clone(),
                            severity: pattern.severity,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Check JSON nesting depth
    fn check_nesting_depth(
        &self,
        value: &serde_json::Value,
        depth: usize,
    ) -> Result<(), ValidationError> {
        if depth > self.config.max_nesting_depth {
            return Err(ValidationError::NestingTooDeep {
                depth,
                max: self.config.max_nesting_depth,
            });
        }

        match value {
            serde_json::Value::Object(map) => {
                for v in map.values() {
                    self.check_nesting_depth(v, depth + 1)?;
                }
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    self.check_nesting_depth(v, depth + 1)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Apply custom validation rule
    fn apply_custom_rule(
        &self,
        params: &serde_json::Map<String, serde_json::Value>,
        rule: &ValidationRule,
    ) -> Result<(), ValidationError> {
        match rule {
            ValidationRule::RequireOneOf(fields) => {
                let has_any = fields.iter().any(|f| params.contains_key(f));
                if !has_any {
                    return Err(ValidationError::RequireOneOf(fields.clone()));
                }
            }
            ValidationRule::MutuallyExclusive(fields) => {
                let present: Vec<_> = fields.iter()
                    .filter(|f| params.contains_key(*f))
                    .collect();
                if present.len() > 1 {
                    return Err(ValidationError::MutuallyExclusive(
                        present.into_iter().cloned().collect()
                    ));
                }
            }
            ValidationRule::ConditionalRequired { if_field, then_required } => {
                if params.contains_key(if_field) {
                    for required in then_required {
                        if !params.contains_key(required) {
                            return Err(ValidationError::ConditionalRequired {
                                condition: if_field.clone(),
                                missing: required.clone(),
                            });
                        }
                    }
                }
            }
            ValidationRule::CustomValidator(_) => {
                // Custom validators would be looked up from a registry
            }
        }

        Ok(())
    }
}

/// Validated request ready for processing
#[derive(Clone, Debug)]
pub struct ValidatedRequest {
    /// Tool name
    pub tool_name: String,

    /// Sanitized parameters
    pub params: serde_json::Value,

    /// Validation timestamp
    pub validation_timestamp: std::time::Instant,
}

/// Validation errors
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Request too large: {size} bytes exceeds {max} byte limit")]
    RequestTooLarge { size: usize, max: usize },

    #[error("Unknown tool: {0}")]
    UnknownTool(String),

    #[error("Missing required parameter: {0}")]
    MissingRequired(String),

    #[error("Invalid type for field '{field}': expected {expected}")]
    InvalidType { field: String, expected: String },

    #[error("String too short for field '{field}': minimum {min} characters")]
    StringTooShort { field: String, min: usize },

    #[error("String too long for field '{field}': maximum {max} characters")]
    StringTooLong { field: String, max: usize },

    #[error("Pattern mismatch for field '{field}': must match {pattern}")]
    PatternMismatch { field: String, pattern: String },

    #[error("Invalid enum value for field '{field}': '{value}' not in {allowed:?}")]
    InvalidEnumValue { field: String, value: String, allowed: Vec<String> },

    #[error("Number too small for field '{field}': minimum {min}")]
    NumberTooSmall { field: String, min: i64 },

    #[error("Number too large for field '{field}': maximum {max}")]
    NumberTooLarge { field: String, max: i64 },

    #[error("Invalid UUID for field '{field}': {value}")]
    InvalidUuid { field: String, value: String },

    #[error("Array too large for field '{field}': {size} elements exceeds {max}")]
    ArrayTooLarge { field: String, size: usize, max: usize },

    #[error("JSON nesting too deep: {depth} levels exceeds {max}")]
    NestingTooDeep { depth: usize, max: usize },

    #[error("Unknown field: {0}")]
    UnknownField(String),

    #[error("Invalid pattern: {0}")]
    InvalidPattern(String),

    #[error("Injection detected: {pattern} (severity: {severity:?})")]
    InjectionDetected { pattern: String, severity: ThreatSeverity },

    #[error("Require at least one of: {0:?}")]
    RequireOneOf(Vec<String>),

    #[error("Fields are mutually exclusive: {0:?}")]
    MutuallyExclusive(Vec<String>),

    #[error("When '{condition}' is present, '{missing}' is required")]
    ConditionalRequired { condition: String, missing: String },

    #[error("Sanitization failed: {0}")]
    SanitizationFailed(String),
}

impl ValidationError {
    /// Get MCP error code for validation error
    pub fn error_code(&self) -> i32 {
        match self {
            Self::RequestTooLarge { .. } => -32700,  // Parse error
            Self::UnknownTool(_) => -32601,  // Method not found
            Self::MissingRequired(_) => -32602,  // Invalid params
            Self::InvalidType { .. } => -32602,
            Self::StringTooShort { .. } => -32602,
            Self::StringTooLong { .. } => -32602,
            Self::PatternMismatch { .. } => -32602,
            Self::InvalidEnumValue { .. } => -32602,
            Self::NumberTooSmall { .. } => -32602,
            Self::NumberTooLarge { .. } => -32602,
            Self::InvalidUuid { .. } => -32602,
            Self::ArrayTooLarge { .. } => -32602,
            Self::NestingTooDeep { .. } => -32602,
            Self::UnknownField(_) => -32602,
            Self::InvalidPattern(_) => -32602,
            Self::InjectionDetected { .. } => -32600,  // Invalid request
            Self::RequireOneOf(_) => -32602,
            Self::MutuallyExclusive(_) => -32602,
            Self::ConditionalRequired { .. } => -32602,
            Self::SanitizationFailed(_) => -32600,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] All MCP requests pass through InputValidator before processing
- [ ] Validation completes in <1ms (P99)
- [ ] Schema validation catches type mismatches
- [ ] Injection patterns detected with >99% accuracy
- [ ] Size limits enforced at all levels
- [ ] Sanitization removes dangerous content
- [ ] Metrics track validation performance

---

#### REQ-MCPH-002: Injection Pattern Detection

**Priority**: Critical
**Description**: The system SHALL detect and block common injection attack patterns.

```rust
/// Default injection patterns for security
pub fn default_injection_patterns() -> Vec<CompiledPattern> {
    vec![
        // Prompt injection patterns
        CompiledPattern {
            name: "prompt_injection_ignore".to_string(),
            regex: Regex::new(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?").unwrap(),
            severity: ThreatSeverity::Critical,
            action: ThreatAction::Reject,
        },
        CompiledPattern {
            name: "prompt_injection_disregard".to_string(),
            regex: Regex::new(r"(?i)disregard\s+(the\s+)?(system\s+)?prompt").unwrap(),
            severity: ThreatSeverity::Critical,
            action: ThreatAction::Reject,
        },
        CompiledPattern {
            name: "prompt_injection_override".to_string(),
            regex: Regex::new(r"(?i)(new\s+instructions?|override)\s*:").unwrap(),
            severity: ThreatSeverity::High,
            action: ThreatAction::Reject,
        },
        CompiledPattern {
            name: "prompt_injection_identity".to_string(),
            regex: Regex::new(r"(?i)you\s+are\s+now\s+a?").unwrap(),
            severity: ThreatSeverity::High,
            action: ThreatAction::Reject,
        },

        // SQL injection patterns
        CompiledPattern {
            name: "sql_injection_union".to_string(),
            regex: Regex::new(r"(?i)\bunion\s+(all\s+)?select\b").unwrap(),
            severity: ThreatSeverity::Critical,
            action: ThreatAction::Reject,
        },
        CompiledPattern {
            name: "sql_injection_drop".to_string(),
            regex: Regex::new(r"(?i)\bdrop\s+(table|database|index)\b").unwrap(),
            severity: ThreatSeverity::Critical,
            action: ThreatAction::Reject,
        },
        CompiledPattern {
            name: "sql_injection_comment".to_string(),
            regex: Regex::new(r"(--|/\*|\*/|;)\s*(drop|delete|update|insert)").unwrap(),
            severity: ThreatSeverity::High,
            action: ThreatAction::Reject,
        },

        // Path traversal patterns
        CompiledPattern {
            name: "path_traversal".to_string(),
            regex: Regex::new(r"\.\.[\\/]").unwrap(),
            severity: ThreatSeverity::High,
            action: ThreatAction::Reject,
        },
        CompiledPattern {
            name: "path_traversal_encoded".to_string(),
            regex: Regex::new(r"(%2e%2e|%252e%252e|\.%2e|%2e\.)[\\/]").unwrap(),
            severity: ThreatSeverity::High,
            action: ThreatAction::Reject,
        },

        // Script injection patterns
        CompiledPattern {
            name: "script_tag".to_string(),
            regex: Regex::new(r"(?i)<script[^>]*>").unwrap(),
            severity: ThreatSeverity::High,
            action: ThreatAction::Sanitize,
        },
        CompiledPattern {
            name: "javascript_protocol".to_string(),
            regex: Regex::new(r"(?i)javascript\s*:").unwrap(),
            severity: ThreatSeverity::Medium,
            action: ThreatAction::Sanitize,
        },
        CompiledPattern {
            name: "event_handler".to_string(),
            regex: Regex::new(r"(?i)\bon(load|error|click|mouse|focus|blur)\s*=").unwrap(),
            severity: ThreatSeverity::Medium,
            action: ThreatAction::Sanitize,
        },

        // Command injection patterns
        CompiledPattern {
            name: "command_injection".to_string(),
            regex: Regex::new(r"[;&|`$]\s*(cat|ls|rm|wget|curl|bash|sh|python|perl)\b").unwrap(),
            severity: ThreatSeverity::Critical,
            action: ThreatAction::Reject,
        },

        // Null byte injection
        CompiledPattern {
            name: "null_byte".to_string(),
            regex: Regex::new(r"\x00|%00").unwrap(),
            severity: ThreatSeverity::High,
            action: ThreatAction::Reject,
        },
    ]
}
```

**Acceptance Criteria**:
- [ ] All listed injection patterns detected
- [ ] Pattern matching completes in <100us
- [ ] Critical patterns always block request
- [ ] Medium patterns sanitize content
- [ ] Logging captures all detection events

---

### 3.2 Rate Limiting

#### REQ-MCPH-003: RateLimiter with Token Bucket

**Priority**: Critical
**Description**: The system SHALL implement rate limiting using the token bucket algorithm with per-session and global limits.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Rate limiter implementing token bucket algorithm with hierarchical limits.
///
/// Provides global server limits, per-session limits, and per-endpoint limits
/// with configurable burst handling and backpressure signaling.
///
/// `Constraint: Rate_Limit_Check < 100us`
pub struct RateLimiter {
    /// Global token bucket
    pub global_bucket: Arc<RwLock<TokenBucket>>,

    /// Per-session token buckets
    pub session_buckets: Arc<RwLock<HashMap<Uuid, TokenBucket>>>,

    /// Per-endpoint limits
    pub endpoint_limits: HashMap<String, EndpointLimit>,

    /// Configuration
    pub config: RateLimiterConfig,

    /// Metrics
    pub metrics: RateLimiterMetrics,

    /// Blocked sessions (temporary bans)
    pub blocked_sessions: Arc<RwLock<HashMap<Uuid, BlockedSession>>>,
}

/// Configuration for rate limiting
#[derive(Clone, Debug)]
pub struct RateLimiterConfig {
    /// Global requests per minute
    pub global_rpm: u32,  // Default: 10000

    /// Per-session requests per minute
    pub session_rpm: u32,  // Default: 100

    /// Burst multiplier (allows burst up to rate * multiplier)
    pub burst_multiplier: f32,  // Default: 2.0

    /// Time window for rate calculation
    pub window_duration: Duration,  // Default: 1 minute

    /// Maximum sessions to track
    pub max_tracked_sessions: usize,  // Default: 10000

    /// Session bucket TTL (remove inactive)
    pub session_bucket_ttl: Duration,  // Default: 10 minutes

    /// Block duration for repeated violations
    pub violation_block_duration: Duration,  // Default: 5 minutes

    /// Violations before blocking
    pub violations_before_block: u32,  // Default: 10
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            global_rpm: 10000,
            session_rpm: 100,
            burst_multiplier: 2.0,
            window_duration: Duration::from_secs(60),
            max_tracked_sessions: 10000,
            session_bucket_ttl: Duration::from_secs(600),
            violation_block_duration: Duration::from_secs(300),
            violations_before_block: 10,
        }
    }
}

/// Token bucket implementation
#[derive(Clone, Debug)]
pub struct TokenBucket {
    /// Maximum tokens (capacity)
    pub capacity: f64,

    /// Current available tokens
    pub tokens: f64,

    /// Refill rate (tokens per second)
    pub refill_rate: f64,

    /// Last refill timestamp
    pub last_refill: Instant,

    /// Violation count
    pub violations: u32,
}

impl TokenBucket {
    /// Create new token bucket
    pub fn new(rpm: u32, burst_multiplier: f32) -> Self {
        let capacity = (rpm as f64) * (burst_multiplier as f64);
        let refill_rate = (rpm as f64) / 60.0;  // Tokens per second

        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
            violations: 0,
        }
    }

    /// Refill tokens based on elapsed time
    pub fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to acquire tokens
    ///
    /// Returns true if tokens were acquired, false if rate limited.
    pub fn try_acquire(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            self.violations += 1;
            false
        }
    }

    /// Get time until tokens available
    pub fn time_until_available(&self, tokens: f64) -> Duration {
        if self.tokens >= tokens {
            Duration::ZERO
        } else {
            let needed = tokens - self.tokens;
            Duration::from_secs_f64(needed / self.refill_rate)
        }
    }
}

/// Per-endpoint rate limit configuration
#[derive(Clone, Debug)]
pub struct EndpointLimit {
    /// Endpoint name (tool name)
    pub endpoint: String,

    /// Requests per minute for this endpoint
    pub rpm: u32,

    /// Cost multiplier (some endpoints cost more)
    pub cost: f64,

    /// Bypass rate limiting (for health checks)
    pub bypass: bool,
}

/// Blocked session record
#[derive(Clone, Debug)]
pub struct BlockedSession {
    /// Session ID
    pub session_id: Uuid,

    /// Block start time
    pub blocked_at: Instant,

    /// Block duration
    pub duration: Duration,

    /// Reason for block
    pub reason: BlockReason,

    /// Violation count at block time
    pub violations: u32,
}

/// Reasons for session blocking
#[derive(Clone, Debug)]
pub enum BlockReason {
    /// Too many rate limit violations
    RateLimitViolations,
    /// Injection attempt detected
    InjectionAttempt,
    /// Authentication failure
    AuthFailure,
    /// Manual administrative block
    Administrative,
}

/// Rate limiter metrics
#[derive(Clone, Default)]
pub struct RateLimiterMetrics {
    /// Total requests checked
    pub total_requests: u64,

    /// Requests allowed
    pub requests_allowed: u64,

    /// Requests rate limited
    pub requests_limited: u64,

    /// Sessions blocked
    pub sessions_blocked: u64,

    /// Average check latency (microseconds)
    pub avg_check_latency_us: f64,

    /// Current active sessions
    pub active_sessions: usize,
}

impl RateLimiter {
    /// Create new rate limiter with configuration
    pub fn new(config: RateLimiterConfig) -> Self {
        let global_bucket = TokenBucket::new(config.global_rpm, config.burst_multiplier);

        Self {
            global_bucket: Arc::new(RwLock::new(global_bucket)),
            session_buckets: Arc::new(RwLock::new(HashMap::new())),
            endpoint_limits: HashMap::new(),
            config,
            metrics: RateLimiterMetrics::default(),
            blocked_sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if request is allowed
    ///
    /// `Constraint: Check_Latency < 100us`
    pub async fn check_request(
        &mut self,
        session_id: &Uuid,
        endpoint: &str,
    ) -> Result<RateLimitResult, RateLimitError> {
        let start = Instant::now();
        self.metrics.total_requests += 1;

        // Check if session is blocked
        {
            let blocked = self.blocked_sessions.read().await;
            if let Some(block) = blocked.get(session_id) {
                if block.blocked_at.elapsed() < block.duration {
                    return Err(RateLimitError::SessionBlocked {
                        until: block.blocked_at + block.duration,
                        reason: block.reason.clone(),
                    });
                }
            }
        }

        // Get endpoint cost
        let cost = self.endpoint_limits
            .get(endpoint)
            .map(|l| l.cost)
            .unwrap_or(1.0);

        // Check if endpoint bypasses rate limiting
        if self.endpoint_limits.get(endpoint).map(|l| l.bypass).unwrap_or(false) {
            self.metrics.requests_allowed += 1;
            return Ok(RateLimitResult::Allowed {
                remaining_global: 0,
                remaining_session: 0,
            });
        }

        // Check global limit
        {
            let mut global = self.global_bucket.write().await;
            if !global.try_acquire(cost) {
                self.metrics.requests_limited += 1;
                let retry_after = global.time_until_available(cost);
                return Err(RateLimitError::GlobalLimitExceeded { retry_after });
            }
        }

        // Check session limit
        {
            let mut sessions = self.session_buckets.write().await;
            let bucket = sessions.entry(*session_id)
                .or_insert_with(|| TokenBucket::new(
                    self.config.session_rpm,
                    self.config.burst_multiplier,
                ));

            if !bucket.try_acquire(cost) {
                self.metrics.requests_limited += 1;
                let retry_after = bucket.time_until_available(cost);

                // Check if should block session
                if bucket.violations >= self.config.violations_before_block {
                    drop(sessions);  // Release lock before blocking
                    self.block_session(session_id, BlockReason::RateLimitViolations).await;
                    self.metrics.sessions_blocked += 1;
                }

                return Err(RateLimitError::SessionLimitExceeded { retry_after });
            }

            self.metrics.active_sessions = sessions.len();
        }

        self.metrics.requests_allowed += 1;

        // Update latency metric
        let elapsed = start.elapsed().as_micros() as f64;
        self.metrics.avg_check_latency_us =
            (self.metrics.avg_check_latency_us * 0.99) + (elapsed * 0.01);

        // Get remaining tokens for response
        let remaining_global = {
            let global = self.global_bucket.read().await;
            global.tokens as u32
        };
        let remaining_session = {
            let sessions = self.session_buckets.read().await;
            sessions.get(session_id).map(|b| b.tokens as u32).unwrap_or(0)
        };

        Ok(RateLimitResult::Allowed {
            remaining_global,
            remaining_session,
        })
    }

    /// Block a session
    pub async fn block_session(&self, session_id: &Uuid, reason: BlockReason) {
        let mut blocked = self.blocked_sessions.write().await;

        // Get violation count
        let violations = {
            let sessions = self.session_buckets.read().await;
            sessions.get(session_id).map(|b| b.violations).unwrap_or(0)
        };

        blocked.insert(*session_id, BlockedSession {
            session_id: *session_id,
            blocked_at: Instant::now(),
            duration: self.config.violation_block_duration,
            reason,
            violations,
        });

        tracing::warn!(
            session_id = %session_id,
            violations = violations,
            "Session blocked for rate limit violations"
        );
    }

    /// Cleanup stale session buckets
    pub async fn cleanup_stale_sessions(&mut self) {
        let mut sessions = self.session_buckets.write().await;
        let ttl = self.config.session_bucket_ttl;

        sessions.retain(|_, bucket| {
            bucket.last_refill.elapsed() < ttl
        });

        // Also cleanup expired blocks
        let mut blocked = self.blocked_sessions.write().await;
        blocked.retain(|_, block| {
            block.blocked_at.elapsed() < block.duration
        });
    }
}

/// Result of rate limit check
#[derive(Clone, Debug)]
pub enum RateLimitResult {
    /// Request is allowed
    Allowed {
        /// Remaining tokens in global bucket
        remaining_global: u32,
        /// Remaining tokens in session bucket
        remaining_session: u32,
    },
}

/// Rate limit errors
#[derive(Debug, Error)]
pub enum RateLimitError {
    #[error("Global rate limit exceeded, retry after {retry_after:?}")]
    GlobalLimitExceeded { retry_after: Duration },

    #[error("Session rate limit exceeded, retry after {retry_after:?}")]
    SessionLimitExceeded { retry_after: Duration },

    #[error("Session blocked until {until:?} due to {reason:?}")]
    SessionBlocked { until: Instant, reason: BlockReason },
}

impl RateLimitError {
    /// Get MCP error code
    pub fn error_code(&self) -> i32 {
        -32429  // Custom: Too Many Requests
    }

    /// Get retry-after duration
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::GlobalLimitExceeded { retry_after } => Some(*retry_after),
            Self::SessionLimitExceeded { retry_after } => Some(*retry_after),
            Self::SessionBlocked { until, .. } => {
                let now = Instant::now();
                if *until > now {
                    Some(*until - now)
                } else {
                    None
                }
            }
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Token bucket algorithm correctly implemented
- [ ] Rate limit check completes in <100us
- [ ] Global limit of 10000 req/min enforced
- [ ] Per-session limit of 100 req/min enforced
- [ ] Burst handling allows 2x momentary spikes
- [ ] Sessions blocked after repeated violations
- [ ] Metrics track all rate limiting events

---

### 3.3 Authentication and Authorization

#### REQ-MCPH-004: AuthManager with Session Tokens

**Priority**: Critical
**Description**: The system SHALL implement session-based authentication with permission levels and expiry management.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use ring::rand::{SecureRandom, SystemRandom};
use ring::digest::{digest, SHA256};

/// Authentication manager for session-based security.
///
/// Manages session tokens, permission levels, expiry, and token refresh.
/// Implements secure token generation and verification.
///
/// `Constraint: Auth_Verification < 500us`
pub struct AuthManager {
    /// Active sessions
    pub sessions: Arc<RwLock<HashMap<String, Session>>>,

    /// Session configuration
    pub config: AuthConfig,

    /// Metrics
    pub metrics: AuthMetrics,

    /// Random number generator for token generation
    rng: SystemRandom,
}

/// Authentication configuration
#[derive(Clone, Debug)]
pub struct AuthConfig {
    /// Session token TTL
    pub session_ttl: Duration,  // Default: 24 hours

    /// Refresh token TTL
    pub refresh_ttl: Duration,  // Default: 7 days

    /// Token length in bytes
    pub token_length: usize,  // Default: 32

    /// Maximum sessions per user
    pub max_sessions_per_user: usize,  // Default: 5

    /// Enable refresh tokens
    pub enable_refresh: bool,  // Default: true

    /// Require re-auth after inactivity
    pub inactivity_timeout: Duration,  // Default: 1 hour

    /// Maximum failed auth attempts before lockout
    pub max_auth_failures: u32,  // Default: 5

    /// Auth failure lockout duration
    pub lockout_duration: Duration,  // Default: 15 minutes
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            session_ttl: Duration::from_secs(24 * 60 * 60),
            refresh_ttl: Duration::from_secs(7 * 24 * 60 * 60),
            token_length: 32,
            max_sessions_per_user: 5,
            enable_refresh: true,
            inactivity_timeout: Duration::from_secs(60 * 60),
            max_auth_failures: 5,
            lockout_duration: Duration::from_secs(15 * 60),
        }
    }
}

/// Session representation
#[derive(Clone, Debug)]
pub struct Session {
    /// Session ID (UUID)
    pub id: Uuid,

    /// Session token (hashed)
    pub token_hash: String,

    /// User ID associated with session
    pub user_id: String,

    /// Permission level
    pub permission_level: PermissionLevel,

    /// Creation time
    pub created_at: SystemTime,

    /// Expiry time
    pub expires_at: SystemTime,

    /// Last activity time
    pub last_activity: Instant,

    /// Refresh token hash (if enabled)
    pub refresh_token_hash: Option<String>,

    /// Refresh token expiry
    pub refresh_expires_at: Option<SystemTime>,

    /// Session metadata
    pub metadata: SessionMetadata,
}

/// Session metadata
#[derive(Clone, Debug)]
pub struct SessionMetadata {
    /// Client user agent
    pub user_agent: Option<String>,

    /// Client IP address
    pub ip_address: Option<String>,

    /// Request count in this session
    pub request_count: u64,

    /// Last endpoint accessed
    pub last_endpoint: Option<String>,
}

/// Permission levels for authorization
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermissionLevel {
    /// Read-only access (search, inject_context)
    ReadOnly = 0,

    /// Standard access (read + write memories)
    Standard = 1,

    /// Admin access (all operations including config)
    Admin = 2,
}

impl PermissionLevel {
    /// Check if level grants access to endpoint
    pub fn can_access(&self, required: PermissionLevel) -> bool {
        *self >= required
    }

    /// Get required level for endpoint
    pub fn required_for_endpoint(endpoint: &str) -> PermissionLevel {
        match endpoint {
            // Read-only endpoints
            "inject_context" | "search_graph" | "get_memetic_status" |
            "get_neighborhood" | "get_recent_context" | "get_graph_manifest" |
            "entailment_query" | "find_causal_path" => PermissionLevel::ReadOnly,

            // Admin endpoints
            "reload_manifest" | "set_verbosity" | "forget_concept" |
            "restore_from_hash" | "homeostatic_status" | "get_system_logs" => PermissionLevel::Admin,

            // Everything else requires standard
            _ => PermissionLevel::Standard,
        }
    }
}

/// Authentication result
#[derive(Clone, Debug)]
pub struct AuthResult {
    /// Session token
    pub token: String,

    /// Refresh token (if enabled)
    pub refresh_token: Option<String>,

    /// Session ID
    pub session_id: Uuid,

    /// Token expiry
    pub expires_at: SystemTime,

    /// Permission level
    pub permission_level: PermissionLevel,
}

/// Authentication metrics
#[derive(Clone, Default)]
pub struct AuthMetrics {
    /// Total authentications
    pub total_auths: u64,

    /// Successful authentications
    pub successful_auths: u64,

    /// Failed authentications
    pub failed_auths: u64,

    /// Sessions created
    pub sessions_created: u64,

    /// Sessions expired
    pub sessions_expired: u64,

    /// Tokens refreshed
    pub tokens_refreshed: u64,

    /// Average verification latency (microseconds)
    pub avg_verify_latency_us: f64,
}

impl AuthManager {
    /// Create new auth manager
    pub fn new(config: AuthConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: AuthMetrics::default(),
            rng: SystemRandom::new(),
        }
    }

    /// Create new session
    pub async fn create_session(
        &mut self,
        user_id: &str,
        permission_level: PermissionLevel,
        metadata: SessionMetadata,
    ) -> Result<AuthResult, AuthError> {
        // Check existing sessions for user
        {
            let sessions = self.sessions.read().await;
            let user_sessions: Vec<_> = sessions.values()
                .filter(|s| s.user_id == user_id)
                .collect();

            if user_sessions.len() >= self.config.max_sessions_per_user {
                return Err(AuthError::TooManySessions {
                    max: self.config.max_sessions_per_user,
                });
            }
        }

        // Generate session token
        let token = self.generate_token()?;
        let token_hash = self.hash_token(&token);

        // Generate refresh token if enabled
        let (refresh_token, refresh_token_hash, refresh_expires_at) = if self.config.enable_refresh {
            let refresh = self.generate_token()?;
            let refresh_hash = self.hash_token(&refresh);
            let refresh_expires = SystemTime::now() + self.config.refresh_ttl;
            (Some(refresh), Some(refresh_hash), Some(refresh_expires))
        } else {
            (None, None, None)
        };

        // Create session
        let session_id = Uuid::new_v4();
        let now = SystemTime::now();
        let expires_at = now + self.config.session_ttl;

        let session = Session {
            id: session_id,
            token_hash,
            user_id: user_id.to_string(),
            permission_level,
            created_at: now,
            expires_at,
            last_activity: Instant::now(),
            refresh_token_hash,
            refresh_expires_at,
            metadata,
        };

        // Store session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(token.clone(), session);
        }

        self.metrics.sessions_created += 1;
        self.metrics.successful_auths += 1;

        Ok(AuthResult {
            token,
            refresh_token,
            session_id,
            expires_at,
            permission_level,
        })
    }

    /// Verify session token
    ///
    /// `Constraint: Verify_Latency < 500us`
    pub async fn verify_token(
        &mut self,
        token: &str,
        required_permission: PermissionLevel,
    ) -> Result<VerifiedSession, AuthError> {
        let start = Instant::now();
        self.metrics.total_auths += 1;

        let sessions = self.sessions.read().await;
        let session = sessions.get(token)
            .ok_or(AuthError::InvalidToken)?;

        // Check expiry
        if SystemTime::now() > session.expires_at {
            drop(sessions);
            self.metrics.sessions_expired += 1;
            return Err(AuthError::TokenExpired);
        }

        // Check inactivity
        if session.last_activity.elapsed() > self.config.inactivity_timeout {
            return Err(AuthError::InactivityTimeout);
        }

        // Check permission level
        if !session.permission_level.can_access(required_permission) {
            self.metrics.failed_auths += 1;
            return Err(AuthError::InsufficientPermissions {
                required: required_permission,
                actual: session.permission_level,
            });
        }

        // Update latency metric
        let elapsed = start.elapsed().as_micros() as f64;
        self.metrics.avg_verify_latency_us =
            (self.metrics.avg_verify_latency_us * 0.99) + (elapsed * 0.01);

        self.metrics.successful_auths += 1;

        Ok(VerifiedSession {
            session_id: session.id,
            user_id: session.user_id.clone(),
            permission_level: session.permission_level,
        })
    }

    /// Update session activity
    pub async fn update_activity(&self, token: &str, endpoint: &str) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(token) {
            session.last_activity = Instant::now();
            session.metadata.request_count += 1;
            session.metadata.last_endpoint = Some(endpoint.to_string());
        }
    }

    /// Refresh session token
    pub async fn refresh_session(
        &mut self,
        refresh_token: &str,
    ) -> Result<AuthResult, AuthError> {
        if !self.config.enable_refresh {
            return Err(AuthError::RefreshDisabled);
        }

        let refresh_hash = self.hash_token(refresh_token);

        // Find session with matching refresh token
        let mut sessions = self.sessions.write().await;
        let (old_token, session) = sessions.iter_mut()
            .find(|(_, s)| {
                s.refresh_token_hash.as_ref() == Some(&refresh_hash)
            })
            .ok_or(AuthError::InvalidRefreshToken)?;

        // Check refresh expiry
        if let Some(expires) = session.refresh_expires_at {
            if SystemTime::now() > expires {
                return Err(AuthError::RefreshTokenExpired);
            }
        }

        // Generate new tokens
        let new_token = self.generate_token()?;
        let new_token_hash = self.hash_token(&new_token);
        let new_refresh = self.generate_token()?;
        let new_refresh_hash = self.hash_token(&new_refresh);

        // Update session
        let now = SystemTime::now();
        session.token_hash = new_token_hash;
        session.expires_at = now + self.config.session_ttl;
        session.refresh_token_hash = Some(new_refresh_hash);
        session.refresh_expires_at = Some(now + self.config.refresh_ttl);
        session.last_activity = Instant::now();

        let result = AuthResult {
            token: new_token.clone(),
            refresh_token: Some(new_refresh),
            session_id: session.id,
            expires_at: session.expires_at,
            permission_level: session.permission_level,
        };

        // Remove old token entry, add new
        let old_token = old_token.clone();
        let session = sessions.remove(&old_token).unwrap();
        sessions.insert(new_token, session);

        self.metrics.tokens_refreshed += 1;

        Ok(result)
    }

    /// Revoke session
    pub async fn revoke_session(&mut self, token: &str) -> Result<(), AuthError> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(token)
            .ok_or(AuthError::InvalidToken)?;
        Ok(())
    }

    /// Generate secure random token
    fn generate_token(&self) -> Result<String, AuthError> {
        let mut bytes = vec![0u8; self.config.token_length];
        self.rng.fill(&mut bytes)
            .map_err(|_| AuthError::TokenGenerationFailed)?;
        Ok(base64::encode_config(&bytes, base64::URL_SAFE_NO_PAD))
    }

    /// Hash token for storage
    fn hash_token(&self, token: &str) -> String {
        let hash = digest(&SHA256, token.as_bytes());
        hex::encode(hash.as_ref())
    }

    /// Cleanup expired sessions
    pub async fn cleanup_expired(&mut self) {
        let now = SystemTime::now();
        let mut sessions = self.sessions.write().await;
        let before = sessions.len();

        sessions.retain(|_, session| {
            now < session.expires_at
        });

        let removed = before - sessions.len();
        self.metrics.sessions_expired += removed as u64;
    }
}

/// Verified session information
#[derive(Clone, Debug)]
pub struct VerifiedSession {
    /// Session ID
    pub session_id: Uuid,

    /// User ID
    pub user_id: String,

    /// Permission level
    pub permission_level: PermissionLevel,
}

/// Authentication errors
#[derive(Debug, Error)]
pub enum AuthError {
    #[error("Invalid token")]
    InvalidToken,

    #[error("Token expired")]
    TokenExpired,

    #[error("Session timed out due to inactivity")]
    InactivityTimeout,

    #[error("Insufficient permissions: required {required:?}, have {actual:?}")]
    InsufficientPermissions {
        required: PermissionLevel,
        actual: PermissionLevel,
    },

    #[error("Too many sessions: maximum {max}")]
    TooManySessions { max: usize },

    #[error("Refresh tokens disabled")]
    RefreshDisabled,

    #[error("Invalid refresh token")]
    InvalidRefreshToken,

    #[error("Refresh token expired")]
    RefreshTokenExpired,

    #[error("Token generation failed")]
    TokenGenerationFailed,

    #[error("Account locked out")]
    AccountLocked,
}

impl AuthError {
    /// Get MCP error code
    pub fn error_code(&self) -> i32 {
        match self {
            Self::InvalidToken | Self::InvalidRefreshToken => -32401,  // Unauthorized
            Self::TokenExpired | Self::RefreshTokenExpired => -32401,
            Self::InactivityTimeout => -32401,
            Self::InsufficientPermissions { .. } => -32403,  // Forbidden
            Self::TooManySessions { .. } => -32429,  // Too many requests
            Self::RefreshDisabled => -32400,  // Bad request
            Self::TokenGenerationFailed => -32500,  // Internal error
            Self::AccountLocked => -32403,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Session tokens generated with cryptographic randomness
- [ ] Token verification completes in <500us
- [ ] Permission levels enforced per endpoint
- [ ] Session expiry at 24 hours default
- [ ] Refresh tokens extend session life
- [ ] Inactivity timeout enforced
- [ ] Metrics track all auth events

---

### 3.4 Audit Logging

#### REQ-MCPH-005: AuditLogger with Tamper-Evident Records

**Priority**: Critical
**Description**: The system SHALL implement structured audit logging with tamper-evident checksums and configurable retention.

```rust
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::mpsc;
use sha2::{Sha256, Digest};

/// Audit logger with tamper-evident, structured logging.
///
/// Records all operations with cryptographic chaining for integrity
/// verification. Supports async non-blocking writes.
///
/// `Constraint: Audit_Write < 200us (async)`
pub struct AuditLogger {
    /// Async channel sender for log entries
    pub sender: mpsc::Sender<AuditEntry>,

    /// Configuration
    pub config: AuditConfig,

    /// Metrics
    pub metrics: Arc<tokio::sync::RwLock<AuditMetrics>>,
}

/// Audit logger worker (runs in background)
pub struct AuditLoggerWorker {
    /// Async channel receiver
    receiver: mpsc::Receiver<AuditEntry>,

    /// Log storage backend
    storage: Box<dyn AuditStorage + Send + Sync>,

    /// Previous entry hash (for chaining)
    prev_hash: String,

    /// Configuration
    config: AuditConfig,

    /// Metrics reference
    metrics: Arc<tokio::sync::RwLock<AuditMetrics>>,
}

/// Audit configuration
#[derive(Clone, Debug)]
pub struct AuditConfig {
    /// Channel buffer size
    pub channel_buffer: usize,  // Default: 10000

    /// Retention period
    pub retention_days: u32,  // Default: 90

    /// Enable tamper-evident chaining
    pub enable_chaining: bool,  // Default: true

    /// Batch write size
    pub batch_size: usize,  // Default: 100

    /// Flush interval
    pub flush_interval: Duration,  // Default: 1 second

    /// Log level filter
    pub min_level: AuditLevel,  // Default: Info

    /// PII masking enabled
    pub mask_pii: bool,  // Default: true
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            channel_buffer: 10000,
            retention_days: 90,
            enable_chaining: true,
            batch_size: 100,
            flush_interval: Duration::from_secs(1),
            min_level: AuditLevel::Info,
            mask_pii: true,
        }
    }
}

/// Audit entry (single log record)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID
    pub id: Uuid,

    /// Entry timestamp
    pub timestamp: SystemTime,

    /// Audit level
    pub level: AuditLevel,

    /// Event type
    pub event_type: AuditEventType,

    /// Session ID (if applicable)
    pub session_id: Option<Uuid>,

    /// User ID (if applicable)
    pub user_id: Option<String>,

    /// Tool/endpoint name
    pub endpoint: String,

    /// Request summary (sanitized)
    pub request_summary: String,

    /// Response status
    pub response_status: ResponseStatus,

    /// Latency in microseconds
    pub latency_us: u64,

    /// Additional context
    pub context: HashMap<String, String>,

    /// Previous entry hash (for chain verification)
    pub prev_hash: Option<String>,

    /// This entry's hash
    pub entry_hash: String,
}

/// Audit levels
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuditLevel {
    /// Debug information
    Debug = 0,
    /// Informational events
    Info = 1,
    /// Warning conditions
    Warn = 2,
    /// Error conditions
    Error = 3,
    /// Security events
    Security = 4,
    /// Critical events
    Critical = 5,
}

/// Audit event types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Authentication event
    Authentication { success: bool },
    /// Authorization check
    Authorization { granted: bool },
    /// Tool invocation
    ToolInvocation,
    /// Rate limit event
    RateLimit { blocked: bool },
    /// Validation event
    Validation { passed: bool },
    /// Security event
    SecurityEvent { threat_type: String },
    /// Configuration change
    ConfigChange { setting: String },
    /// Data access
    DataAccess { operation: String },
    /// System event
    System { operation: String },
}

/// Response status
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResponseStatus {
    /// Successful response
    Success,
    /// Client error (4xx equivalent)
    ClientError { code: i32, message: String },
    /// Server error (5xx equivalent)
    ServerError { code: i32, message: String },
    /// Rate limited
    RateLimited,
}

/// Audit metrics
#[derive(Clone, Default)]
pub struct AuditMetrics {
    /// Total entries logged
    pub total_entries: u64,

    /// Entries by level
    pub entries_by_level: HashMap<String, u64>,

    /// Entries dropped (channel full)
    pub entries_dropped: u64,

    /// Average write latency (microseconds)
    pub avg_write_latency_us: f64,

    /// Storage size bytes
    pub storage_size_bytes: u64,

    /// Chain verification failures
    pub chain_failures: u64,
}

/// Audit storage trait
#[async_trait::async_trait]
pub trait AuditStorage {
    /// Write batch of entries
    async fn write_batch(&mut self, entries: Vec<AuditEntry>) -> Result<(), AuditError>;

    /// Query entries by time range
    async fn query(
        &self,
        start: SystemTime,
        end: SystemTime,
        filters: AuditFilters,
    ) -> Result<Vec<AuditEntry>, AuditError>;

    /// Verify chain integrity
    async fn verify_chain(&self, start: SystemTime, end: SystemTime) -> Result<bool, AuditError>;

    /// Get storage size
    async fn storage_size(&self) -> u64;

    /// Cleanup old entries
    async fn cleanup(&mut self, before: SystemTime) -> Result<u64, AuditError>;
}

/// Audit query filters
#[derive(Clone, Debug, Default)]
pub struct AuditFilters {
    pub level: Option<AuditLevel>,
    pub event_type: Option<String>,
    pub session_id: Option<Uuid>,
    pub user_id: Option<String>,
    pub endpoint: Option<String>,
    pub limit: Option<usize>,
}

impl AuditLogger {
    /// Create new audit logger
    pub fn new(config: AuditConfig) -> (Self, AuditLoggerWorker) {
        let (sender, receiver) = mpsc::channel(config.channel_buffer);
        let metrics = Arc::new(tokio::sync::RwLock::new(AuditMetrics::default()));

        let logger = Self {
            sender,
            config: config.clone(),
            metrics: metrics.clone(),
        };

        let worker = AuditLoggerWorker {
            receiver,
            storage: Box::new(InMemoryAuditStorage::new()),
            prev_hash: "genesis".to_string(),
            config,
            metrics,
        };

        (logger, worker)
    }

    /// Log an audit entry (async, non-blocking)
    ///
    /// `Constraint: Log_Latency < 200us`
    pub async fn log(&self, mut entry: AuditEntry) {
        // Filter by level
        if entry.level < self.config.min_level {
            return;
        }

        // Mask PII if enabled
        if self.config.mask_pii {
            entry = self.mask_pii(entry);
        }

        // Try to send (non-blocking)
        match self.sender.try_send(entry) {
            Ok(_) => {
                // Update metrics asynchronously
                let metrics = self.metrics.clone();
                tokio::spawn(async move {
                    let mut m = metrics.write().await;
                    m.total_entries += 1;
                });
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                // Channel full, drop entry and record
                let metrics = self.metrics.clone();
                tokio::spawn(async move {
                    let mut m = metrics.write().await;
                    m.entries_dropped += 1;
                });
            }
            Err(_) => {
                // Channel closed, nothing to do
            }
        }
    }

    /// Create builder for audit entry
    pub fn entry(&self, event_type: AuditEventType) -> AuditEntryBuilder {
        AuditEntryBuilder::new(event_type)
    }

    /// Mask PII in entry
    fn mask_pii(&self, mut entry: AuditEntry) -> AuditEntry {
        // Mask common PII patterns
        let pii_patterns = [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
            (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]"),
            (r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+", "[REDACTED]"),
        ];

        for (pattern, replacement) in pii_patterns {
            if let Ok(re) = Regex::new(pattern) {
                entry.request_summary = re.replace_all(&entry.request_summary, replacement).to_string();
            }
        }

        entry
    }
}

impl AuditLoggerWorker {
    /// Run the worker (background task)
    pub async fn run(mut self) {
        let mut batch: Vec<AuditEntry> = Vec::with_capacity(self.config.batch_size);
        let mut flush_interval = tokio::time::interval(self.config.flush_interval);

        loop {
            tokio::select! {
                // Receive entries
                Some(mut entry) = self.receiver.recv() => {
                    // Add chain hash if enabled
                    if self.config.enable_chaining {
                        entry.prev_hash = Some(self.prev_hash.clone());
                        entry.entry_hash = self.compute_hash(&entry);
                        self.prev_hash = entry.entry_hash.clone();
                    } else {
                        entry.entry_hash = self.compute_hash(&entry);
                    }

                    batch.push(entry);

                    // Flush if batch full
                    if batch.len() >= self.config.batch_size {
                        self.flush_batch(&mut batch).await;
                    }
                }

                // Periodic flush
                _ = flush_interval.tick() => {
                    if !batch.is_empty() {
                        self.flush_batch(&mut batch).await;
                    }
                }
            }
        }
    }

    /// Flush batch to storage
    async fn flush_batch(&mut self, batch: &mut Vec<AuditEntry>) {
        let start = std::time::Instant::now();
        let entries: Vec<_> = batch.drain(..).collect();
        let count = entries.len();

        if let Err(e) = self.storage.write_batch(entries).await {
            tracing::error!(error = %e, "Failed to write audit batch");
        }

        // Update metrics
        let elapsed = start.elapsed().as_micros() as f64 / count as f64;
        let mut metrics = self.metrics.write().await;
        metrics.avg_write_latency_us =
            (metrics.avg_write_latency_us * 0.9) + (elapsed * 0.1);
        metrics.storage_size_bytes = self.storage.storage_size().await;
    }

    /// Compute entry hash
    fn compute_hash(&self, entry: &AuditEntry) -> String {
        let mut hasher = Sha256::new();

        // Include key fields in hash
        hasher.update(entry.id.to_string().as_bytes());
        hasher.update(format!("{:?}", entry.timestamp).as_bytes());
        hasher.update(entry.endpoint.as_bytes());
        hasher.update(entry.request_summary.as_bytes());
        if let Some(ref prev) = entry.prev_hash {
            hasher.update(prev.as_bytes());
        }

        hex::encode(hasher.finalize())
    }
}

/// Builder for audit entries
pub struct AuditEntryBuilder {
    event_type: AuditEventType,
    level: AuditLevel,
    session_id: Option<Uuid>,
    user_id: Option<String>,
    endpoint: String,
    request_summary: String,
    response_status: ResponseStatus,
    latency_us: u64,
    context: HashMap<String, String>,
}

impl AuditEntryBuilder {
    pub fn new(event_type: AuditEventType) -> Self {
        Self {
            event_type,
            level: AuditLevel::Info,
            session_id: None,
            user_id: None,
            endpoint: String::new(),
            request_summary: String::new(),
            response_status: ResponseStatus::Success,
            latency_us: 0,
            context: HashMap::new(),
        }
    }

    pub fn level(mut self, level: AuditLevel) -> Self {
        self.level = level;
        self
    }

    pub fn session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn user(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = endpoint.to_string();
        self
    }

    pub fn request(mut self, summary: &str) -> Self {
        self.request_summary = summary.to_string();
        self
    }

    pub fn status(mut self, status: ResponseStatus) -> Self {
        self.response_status = status;
        self
    }

    pub fn latency(mut self, latency_us: u64) -> Self {
        self.latency_us = latency_us;
        self
    }

    pub fn context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    pub fn build(self) -> AuditEntry {
        AuditEntry {
            id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            level: self.level,
            event_type: self.event_type,
            session_id: self.session_id,
            user_id: self.user_id,
            endpoint: self.endpoint,
            request_summary: self.request_summary,
            response_status: self.response_status,
            latency_us: self.latency_us,
            context: self.context,
            prev_hash: None,
            entry_hash: String::new(),
        }
    }
}

/// In-memory audit storage (for testing/development)
struct InMemoryAuditStorage {
    entries: VecDeque<AuditEntry>,
    max_entries: usize,
}

impl InMemoryAuditStorage {
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: 100000,
        }
    }
}

#[async_trait::async_trait]
impl AuditStorage for InMemoryAuditStorage {
    async fn write_batch(&mut self, entries: Vec<AuditEntry>) -> Result<(), AuditError> {
        for entry in entries {
            if self.entries.len() >= self.max_entries {
                self.entries.pop_front();
            }
            self.entries.push_back(entry);
        }
        Ok(())
    }

    async fn query(
        &self,
        start: SystemTime,
        end: SystemTime,
        filters: AuditFilters,
    ) -> Result<Vec<AuditEntry>, AuditError> {
        let mut results: Vec<_> = self.entries.iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .filter(|e| {
                if let Some(level) = filters.level {
                    e.level >= level
                } else {
                    true
                }
            })
            .filter(|e| {
                if let Some(ref session) = filters.session_id {
                    e.session_id.as_ref() == Some(session)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        if let Some(limit) = filters.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn verify_chain(&self, _start: SystemTime, _end: SystemTime) -> Result<bool, AuditError> {
        // Simple chain verification
        let mut prev_hash = "genesis".to_string();

        for entry in &self.entries {
            if let Some(ref entry_prev) = entry.prev_hash {
                if *entry_prev != prev_hash {
                    return Ok(false);
                }
            }
            prev_hash = entry.entry_hash.clone();
        }

        Ok(true)
    }

    async fn storage_size(&self) -> u64 {
        // Approximate size
        (self.entries.len() * 1024) as u64
    }

    async fn cleanup(&mut self, before: SystemTime) -> Result<u64, AuditError> {
        let before_len = self.entries.len();
        self.entries.retain(|e| e.timestamp >= before);
        Ok((before_len - self.entries.len()) as u64)
    }
}

/// Audit errors
#[derive(Debug, Error)]
pub enum AuditError {
    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Chain verification failed")]
    ChainVerificationFailed,

    #[error("Query error: {0}")]
    QueryError(String),
}
```

**Acceptance Criteria**:
- [ ] Async non-blocking audit writes <200us
- [ ] Tamper-evident hash chaining
- [ ] PII masking in log entries
- [ ] Configurable retention policies
- [ ] Batch writing for efficiency
- [ ] Chain integrity verification
- [ ] Query support with filters

---

### 3.5 Resource Quota Management

#### REQ-MCPH-006: ResourceQuotaManager

**Priority**: High
**Description**: The system SHALL enforce resource quotas for memory, CPU time, and storage per session.

```rust
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Resource quota manager for enforcing per-session limits.
///
/// Tracks and enforces memory, CPU time, and storage quotas
/// to prevent resource exhaustion attacks.
pub struct ResourceQuotaManager {
    /// Per-session resource usage
    pub session_usage: Arc<RwLock<HashMap<Uuid, SessionResourceUsage>>>,

    /// Configuration
    pub config: QuotaConfig,

    /// Metrics
    pub metrics: QuotaMetrics,
}

/// Quota configuration
#[derive(Clone, Debug)]
pub struct QuotaConfig {
    /// Maximum memory per session (bytes)
    pub max_memory_per_session: usize,  // Default: 100MB

    /// Maximum CPU time per request (microseconds)
    pub max_cpu_time_per_request: u64,  // Default: 5000 (5ms)

    /// Maximum storage per session (bytes)
    pub max_storage_per_session: usize,  // Default: 1GB

    /// Maximum concurrent requests per session
    pub max_concurrent_requests: u32,  // Default: 10

    /// Maximum request rate (per second)
    pub max_request_rate: f64,  // Default: 100.0

    /// Quota check interval
    pub check_interval: Duration,  // Default: 100ms

    /// Grace period before hard enforcement
    pub grace_period: Duration,  // Default: 5 seconds
}

impl Default for QuotaConfig {
    fn default() -> Self {
        Self {
            max_memory_per_session: 100 * 1024 * 1024,  // 100MB
            max_cpu_time_per_request: 5000,  // 5ms
            max_storage_per_session: 1024 * 1024 * 1024,  // 1GB
            max_concurrent_requests: 10,
            max_request_rate: 100.0,
            check_interval: Duration::from_millis(100),
            grace_period: Duration::from_secs(5),
        }
    }
}

/// Per-session resource usage tracking
#[derive(Clone, Debug)]
pub struct SessionResourceUsage {
    /// Session ID
    pub session_id: Uuid,

    /// Current memory usage (bytes)
    pub memory_used: usize,

    /// Total CPU time used (microseconds)
    pub cpu_time_used: u64,

    /// Storage used (bytes)
    pub storage_used: usize,

    /// Current concurrent requests
    pub concurrent_requests: u32,

    /// Request timestamps for rate calculation
    pub request_timestamps: Vec<Instant>,

    /// Last updated
    pub last_updated: Instant,

    /// Quota violations count
    pub violations: u32,

    /// In grace period
    pub in_grace_period: bool,

    /// Grace period start
    pub grace_period_start: Option<Instant>,
}

impl SessionResourceUsage {
    pub fn new(session_id: Uuid) -> Self {
        Self {
            session_id,
            memory_used: 0,
            cpu_time_used: 0,
            storage_used: 0,
            concurrent_requests: 0,
            request_timestamps: Vec::new(),
            last_updated: Instant::now(),
            violations: 0,
            in_grace_period: false,
            grace_period_start: None,
        }
    }

    /// Calculate current request rate
    pub fn current_rate(&self) -> f64 {
        let now = Instant::now();
        let recent: Vec<_> = self.request_timestamps.iter()
            .filter(|&&t| now.duration_since(t) < Duration::from_secs(1))
            .collect();
        recent.len() as f64
    }
}

/// Quota metrics
#[derive(Clone, Default)]
pub struct QuotaMetrics {
    /// Total quota checks
    pub total_checks: u64,

    /// Violations by type
    pub violations_by_type: HashMap<String, u64>,

    /// Sessions in grace period
    pub sessions_in_grace: u64,

    /// Sessions hard-limited
    pub sessions_limited: u64,
}

impl ResourceQuotaManager {
    /// Create new quota manager
    pub fn new(config: QuotaConfig) -> Self {
        Self {
            session_usage: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: QuotaMetrics::default(),
        }
    }

    /// Check if request is within quota
    pub async fn check_quota(
        &mut self,
        session_id: &Uuid,
        estimated_memory: usize,
        estimated_cpu: u64,
    ) -> Result<QuotaCheckResult, QuotaError> {
        self.metrics.total_checks += 1;

        let mut usage_map = self.session_usage.write().await;
        let usage = usage_map.entry(*session_id)
            .or_insert_with(|| SessionResourceUsage::new(*session_id));

        // Record request timestamp
        let now = Instant::now();
        usage.request_timestamps.push(now);
        usage.request_timestamps.retain(|&t| now.duration_since(t) < Duration::from_secs(60));

        // Check concurrent requests
        if usage.concurrent_requests >= self.config.max_concurrent_requests {
            usage.violations += 1;
            self.record_violation("concurrent_requests");
            return Err(QuotaError::TooManyConcurrentRequests {
                current: usage.concurrent_requests,
                max: self.config.max_concurrent_requests,
            });
        }

        // Check request rate
        let rate = usage.current_rate();
        if rate > self.config.max_request_rate {
            usage.violations += 1;
            self.record_violation("request_rate");
            return Err(QuotaError::RateLimitExceeded {
                current_rate: rate,
                max_rate: self.config.max_request_rate,
            });
        }

        // Check memory quota
        let projected_memory = usage.memory_used + estimated_memory;
        if projected_memory > self.config.max_memory_per_session {
            usage.violations += 1;
            self.record_violation("memory");

            // Check grace period
            if !usage.in_grace_period {
                usage.in_grace_period = true;
                usage.grace_period_start = Some(now);
                return Ok(QuotaCheckResult::GracePeriod {
                    resource: "memory".to_string(),
                    current: projected_memory,
                    max: self.config.max_memory_per_session,
                    remaining: self.config.grace_period,
                });
            } else if let Some(start) = usage.grace_period_start {
                if now.duration_since(start) > self.config.grace_period {
                    return Err(QuotaError::MemoryQuotaExceeded {
                        used: projected_memory,
                        max: self.config.max_memory_per_session,
                    });
                }
            }
        }

        // Check CPU quota
        if estimated_cpu > self.config.max_cpu_time_per_request {
            usage.violations += 1;
            self.record_violation("cpu_time");
            return Err(QuotaError::CpuQuotaExceeded {
                estimated: estimated_cpu,
                max: self.config.max_cpu_time_per_request,
            });
        }

        // Increment concurrent requests
        usage.concurrent_requests += 1;
        usage.last_updated = now;

        Ok(QuotaCheckResult::Allowed {
            memory_remaining: self.config.max_memory_per_session.saturating_sub(usage.memory_used),
            cpu_remaining: self.config.max_cpu_time_per_request,
        })
    }

    /// Record request completion
    pub async fn record_completion(
        &self,
        session_id: &Uuid,
        memory_used: usize,
        cpu_used: u64,
    ) {
        let mut usage_map = self.session_usage.write().await;
        if let Some(usage) = usage_map.get_mut(session_id) {
            usage.concurrent_requests = usage.concurrent_requests.saturating_sub(1);
            usage.memory_used += memory_used;
            usage.cpu_time_used += cpu_used;
            usage.last_updated = Instant::now();
        }
    }

    /// Record storage usage
    pub async fn record_storage(
        &self,
        session_id: &Uuid,
        storage_delta: i64,
    ) -> Result<(), QuotaError> {
        let mut usage_map = self.session_usage.write().await;
        let usage = usage_map.entry(*session_id)
            .or_insert_with(|| SessionResourceUsage::new(*session_id));

        let new_storage = if storage_delta >= 0 {
            usage.storage_used.saturating_add(storage_delta as usize)
        } else {
            usage.storage_used.saturating_sub((-storage_delta) as usize)
        };

        if new_storage > self.config.max_storage_per_session {
            return Err(QuotaError::StorageQuotaExceeded {
                used: new_storage,
                max: self.config.max_storage_per_session,
            });
        }

        usage.storage_used = new_storage;
        Ok(())
    }

    /// Get session usage
    pub async fn get_usage(&self, session_id: &Uuid) -> Option<SessionResourceUsage> {
        let usage_map = self.session_usage.read().await;
        usage_map.get(session_id).cloned()
    }

    /// Cleanup stale sessions
    pub async fn cleanup_stale(&mut self, max_age: Duration) {
        let now = Instant::now();
        let mut usage_map = self.session_usage.write().await;
        usage_map.retain(|_, usage| {
            now.duration_since(usage.last_updated) < max_age
        });
    }

    fn record_violation(&mut self, violation_type: &str) {
        *self.metrics.violations_by_type
            .entry(violation_type.to_string())
            .or_insert(0) += 1;
    }
}

/// Result of quota check
#[derive(Clone, Debug)]
pub enum QuotaCheckResult {
    /// Request allowed
    Allowed {
        memory_remaining: usize,
        cpu_remaining: u64,
    },
    /// In grace period (soft limit exceeded)
    GracePeriod {
        resource: String,
        current: usize,
        max: usize,
        remaining: Duration,
    },
}

/// Quota errors
#[derive(Debug, Error)]
pub enum QuotaError {
    #[error("Memory quota exceeded: {used} / {max} bytes")]
    MemoryQuotaExceeded { used: usize, max: usize },

    #[error("CPU quota exceeded: {estimated}us estimated, {max}us max")]
    CpuQuotaExceeded { estimated: u64, max: u64 },

    #[error("Storage quota exceeded: {used} / {max} bytes")]
    StorageQuotaExceeded { used: usize, max: usize },

    #[error("Too many concurrent requests: {current} / {max}")]
    TooManyConcurrentRequests { current: u32, max: u32 },

    #[error("Rate limit exceeded: {current_rate:.1} / {max_rate:.1} req/s")]
    RateLimitExceeded { current_rate: f64, max_rate: f64 },
}

impl QuotaError {
    pub fn error_code(&self) -> i32 {
        match self {
            Self::MemoryQuotaExceeded { .. } => -32507,
            Self::CpuQuotaExceeded { .. } => -32508,
            Self::StorageQuotaExceeded { .. } => -32509,
            Self::TooManyConcurrentRequests { .. } => -32429,
            Self::RateLimitExceeded { .. } => -32429,
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Memory quota enforced per session (100MB default)
- [ ] CPU time quota enforced per request (5ms default)
- [ ] Storage quota enforced per session (1GB default)
- [ ] Concurrent request limit enforced
- [ ] Grace period before hard enforcement
- [ ] Violations tracked and reported

---

### 3.6 Security Middleware Integration

#### REQ-MCPH-007: SecurityMiddleware with Tower Integration

**Priority**: High
**Description**: The system SHALL implement a unified security middleware using the tower pattern for request processing.

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tower::{Layer, Service};

/// Security middleware layer for tower integration.
///
/// Combines all security components into a single middleware layer
/// that processes requests through validation, rate limiting,
/// authentication, quota checking, and audit logging.
pub struct SecurityLayer {
    /// Input validator
    pub validator: Arc<RwLock<InputValidator>>,

    /// Rate limiter
    pub rate_limiter: Arc<RwLock<RateLimiter>>,

    /// Auth manager
    pub auth_manager: Arc<RwLock<AuthManager>>,

    /// Quota manager
    pub quota_manager: Arc<RwLock<ResourceQuotaManager>>,

    /// Audit logger
    pub audit_logger: AuditLogger,
}

impl<S> Layer<S> for SecurityLayer {
    type Service = SecurityMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        SecurityMiddleware {
            inner,
            validator: self.validator.clone(),
            rate_limiter: self.rate_limiter.clone(),
            auth_manager: self.auth_manager.clone(),
            quota_manager: self.quota_manager.clone(),
            audit_logger: self.audit_logger.clone(),
        }
    }
}

/// Security middleware service
pub struct SecurityMiddleware<S> {
    inner: S,
    validator: Arc<RwLock<InputValidator>>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    auth_manager: Arc<RwLock<AuthManager>>,
    quota_manager: Arc<RwLock<ResourceQuotaManager>>,
    audit_logger: AuditLogger,
}

/// MCP request wrapper
#[derive(Clone, Debug)]
pub struct McpRequest {
    /// Request ID
    pub id: serde_json::Value,

    /// Method (tool) name
    pub method: String,

    /// Parameters
    pub params: serde_json::Value,

    /// Session token (from transport)
    pub session_token: Option<String>,
}

/// MCP response wrapper
#[derive(Clone, Debug, Serialize)]
pub struct McpResponse {
    /// JSON-RPC version
    pub jsonrpc: String,

    /// Request ID
    pub id: serde_json::Value,

    /// Result (success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,

    /// Error (failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

/// MCP error response
#[derive(Clone, Debug, Serialize)]
pub struct McpError {
    /// Error code
    pub code: i32,

    /// Error message (sanitized)
    pub message: String,

    /// Additional data (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl<S> Service<McpRequest> for SecurityMiddleware<S>
where
    S: Service<McpRequest, Response = McpResponse> + Clone + Send + 'static,
    S::Future: Send,
{
    type Response = McpResponse;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: McpRequest) -> Self::Future {
        let start = Instant::now();
        let mut inner = self.inner.clone();
        let validator = self.validator.clone();
        let rate_limiter = self.rate_limiter.clone();
        let auth_manager = self.auth_manager.clone();
        let quota_manager = self.quota_manager.clone();
        let audit_logger = self.audit_logger.clone();

        Box::pin(async move {
            let request_id = request.id.clone();
            let method = request.method.clone();

            // Step 1: Validate input
            let validated = {
                let mut v = validator.write().await;
                match v.validate_request(&request.method, &request.params) {
                    Ok(validated) => validated,
                    Err(e) => {
                        audit_logger.log(
                            audit_logger.entry(AuditEventType::Validation { passed: false })
                                .level(AuditLevel::Warn)
                                .endpoint(&method)
                                .request(&format!("Validation failed: {}", e))
                                .status(ResponseStatus::ClientError {
                                    code: e.error_code(),
                                    message: e.to_string(),
                                })
                                .latency(start.elapsed().as_micros() as u64)
                                .build()
                        ).await;

                        return Ok(McpResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request_id,
                            result: None,
                            error: Some(McpError {
                                code: e.error_code(),
                                message: sanitize_error_message(&e.to_string()),
                                data: None,
                            }),
                        });
                    }
                }
            };

            // Step 2: Check rate limit
            let session_id = if let Some(ref token) = request.session_token {
                // Extract session ID from token
                let auth = auth_manager.read().await;
                // Simplified: in practice, verify token and get session
                Uuid::new_v4()  // Placeholder
            } else {
                Uuid::nil()  // Anonymous session
            };

            {
                let mut rl = rate_limiter.write().await;
                match rl.check_request(&session_id, &method).await {
                    Ok(_) => {}
                    Err(e) => {
                        audit_logger.log(
                            audit_logger.entry(AuditEventType::RateLimit { blocked: true })
                                .level(AuditLevel::Warn)
                                .session(session_id)
                                .endpoint(&method)
                                .status(ResponseStatus::RateLimited)
                                .latency(start.elapsed().as_micros() as u64)
                                .build()
                        ).await;

                        return Ok(McpResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request_id,
                            result: None,
                            error: Some(McpError {
                                code: e.error_code(),
                                message: sanitize_error_message(&e.to_string()),
                                data: e.retry_after().map(|d| {
                                    serde_json::json!({ "retry_after_ms": d.as_millis() })
                                }),
                            }),
                        });
                    }
                }
            }

            // Step 3: Authenticate (if token provided)
            let verified_session = if let Some(ref token) = request.session_token {
                let required_permission = PermissionLevel::required_for_endpoint(&method);
                let mut am = auth_manager.write().await;
                match am.verify_token(token, required_permission).await {
                    Ok(verified) => Some(verified),
                    Err(e) => {
                        audit_logger.log(
                            audit_logger.entry(AuditEventType::Authentication { success: false })
                                .level(AuditLevel::Security)
                                .endpoint(&method)
                                .status(ResponseStatus::ClientError {
                                    code: e.error_code(),
                                    message: e.to_string(),
                                })
                                .latency(start.elapsed().as_micros() as u64)
                                .build()
                        ).await;

                        return Ok(McpResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request_id,
                            result: None,
                            error: Some(McpError {
                                code: e.error_code(),
                                message: sanitize_error_message(&e.to_string()),
                                data: None,
                            }),
                        });
                    }
                }
            } else {
                // Check if endpoint allows anonymous access
                let required = PermissionLevel::required_for_endpoint(&method);
                if required > PermissionLevel::ReadOnly {
                    return Ok(McpResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request_id,
                        result: None,
                        error: Some(McpError {
                            code: -32401,
                            message: "Authentication required".to_string(),
                            data: None,
                        }),
                    });
                }
                None
            };

            // Step 4: Check quota
            {
                let estimated_memory = estimate_memory_usage(&validated.params);
                let estimated_cpu = estimate_cpu_time(&method);
                let mut qm = quota_manager.write().await;
                match qm.check_quota(&session_id, estimated_memory, estimated_cpu).await {
                    Ok(_) => {}
                    Err(e) => {
                        audit_logger.log(
                            audit_logger.entry(AuditEventType::System { operation: "quota_check".to_string() })
                                .level(AuditLevel::Warn)
                                .session(session_id)
                                .endpoint(&method)
                                .status(ResponseStatus::ClientError {
                                    code: e.error_code(),
                                    message: e.to_string(),
                                })
                                .latency(start.elapsed().as_micros() as u64)
                                .build()
                        ).await;

                        return Ok(McpResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request_id,
                            result: None,
                            error: Some(McpError {
                                code: e.error_code(),
                                message: sanitize_error_message(&e.to_string()),
                                data: None,
                            }),
                        });
                    }
                }
            }

            // Step 5: Call inner service
            let result = inner.call(request).await;

            // Step 6: Audit successful request
            let latency = start.elapsed().as_micros() as u64;
            match &result {
                Ok(response) => {
                    let status = if response.error.is_some() {
                        ResponseStatus::ClientError {
                            code: response.error.as_ref().map(|e| e.code).unwrap_or(-32600),
                            message: response.error.as_ref()
                                .map(|e| e.message.clone())
                                .unwrap_or_default(),
                        }
                    } else {
                        ResponseStatus::Success
                    };

                    audit_logger.log(
                        audit_logger.entry(AuditEventType::ToolInvocation)
                            .level(AuditLevel::Info)
                            .session(session_id)
                            .user(verified_session.as_ref().map(|s| s.user_id.as_str()).unwrap_or("anonymous"))
                            .endpoint(&method)
                            .request(&format!("Tool invocation: {}", method))
                            .status(status)
                            .latency(latency)
                            .build()
                    ).await;
                }
                Err(_) => {
                    audit_logger.log(
                        audit_logger.entry(AuditEventType::ToolInvocation)
                            .level(AuditLevel::Error)
                            .session(session_id)
                            .endpoint(&method)
                            .status(ResponseStatus::ServerError {
                                code: -32603,
                                message: "Internal error".to_string(),
                            })
                            .latency(latency)
                            .build()
                    ).await;
                }
            }

            // Record completion for quota tracking
            {
                let qm = quota_manager.read().await;
                qm.record_completion(&session_id, 0, latency).await;
            }

            result
        })
    }
}

/// Sanitize error message to prevent information leakage
fn sanitize_error_message(message: &str) -> String {
    // Remove file paths
    let sanitized = regex::Regex::new(r"(/[^\s:]+)+")
        .map(|re| re.replace_all(message, "[path]"))
        .unwrap_or_else(|_| std::borrow::Cow::Borrowed(message));

    // Remove stack traces
    let sanitized = regex::Regex::new(r"at \S+ \(\S+:\d+:\d+\)")
        .map(|re| re.replace_all(&sanitized, ""))
        .unwrap_or(sanitized);

    // Remove internal function names
    let sanitized = regex::Regex::new(r"context_graph::\S+")
        .map(|re| re.replace_all(&sanitized, "[internal]"))
        .unwrap_or(sanitized);

    // Truncate long messages
    if sanitized.len() > 200 {
        format!("{}...", &sanitized[..200])
    } else {
        sanitized.to_string()
    }
}

/// Estimate memory usage for request
fn estimate_memory_usage(params: &serde_json::Value) -> usize {
    // Rough estimate based on JSON size
    let json_size = params.to_string().len();
    // Multiply by 3 for processing overhead
    json_size * 3
}

/// Estimate CPU time for endpoint
fn estimate_cpu_time(endpoint: &str) -> u64 {
    match endpoint {
        "inject_context" => 2000,  // 2ms
        "search_graph" => 1000,    // 1ms
        "store_memory" => 500,     // 0.5ms
        "trigger_dream" => 5000,   // 5ms
        "merge_concepts" => 3000,  // 3ms
        _ => 1000,                 // 1ms default
    }
}
```

**Acceptance Criteria**:
- [ ] Tower middleware integration complete
- [ ] All security layers execute in order
- [ ] Error messages sanitized (no stack traces)
- [ ] Audit logging for all requests
- [ ] Graceful error responses
- [ ] Performance metrics tracked

---

### 3.7 Marblestone MCP Tools

The following MCP tools provide access to the Marblestone Cognitive Architecture components, including steering feedback, omnidirectional inference, and formal verification.

#### REQ-MCPH-008: GetSteeringFeedback MCP Tool

**Priority**: Must
**Description**: The system SHALL expose a get_steering_feedback MCP tool that provides access to the Steering Subsystem (Module 12.5) for evaluating thoughts and memories.

```rust
// ============================================
// MARBLESTONE MCP TOOLS
// ============================================

/// MCP Tool: Get steering feedback for a thought/memory
///
/// Returns SteeringReward from the Steering Subsystem (Module 12.5)
#[mcp_tool]
pub struct GetSteeringFeedback {
    /// Content to evaluate
    pub content: String,
    /// Optional context for evaluation
    pub context: Option<String>,
    /// Domain hint
    pub domain: Option<Domain>,
}

#[mcp_response]
pub struct SteeringFeedbackResponse {
    pub reward: f32,
    pub gardener_score: f32,
    pub curator_score: f32,
    pub assessor_score: f32,
    pub explanation: String,
    pub suggestions: Vec<String>,
}

impl GetSteeringFeedback {
    /// Process steering feedback request
    pub async fn execute(
        &self,
        steering: &SteeringSubsystem,
    ) -> Result<SteeringFeedbackResponse, McpHardeningError> {
        // Validate content length
        if self.content.len() > 65536 {
            return Err(McpHardeningError::Validation(
                ValidationError::StringTooLong {
                    field: "content".to_string(),
                    max_length: 65536,
                    actual_length: self.content.len(),
                }
            ));
        }

        // Get steering reward from Module 12.5
        let reward = steering.evaluate(
            &self.content,
            self.context.as_deref(),
            self.domain.clone(),
        ).await?;

        Ok(SteeringFeedbackResponse {
            reward: reward.total_reward,
            gardener_score: reward.gardener_score,
            curator_score: reward.curator_score,
            assessor_score: reward.assessor_score,
            explanation: reward.explanation,
            suggestions: reward.suggestions,
        })
    }
}
```

**Acceptance Criteria**:
- [ ] Steering feedback accessible via MCP
- [ ] Content validation enforced (max 64KB)
- [ ] Domain hints propagated to steering subsystem
- [ ] Response includes all steering component scores
- [ ] Error handling for steering subsystem failures

---

#### REQ-MCPH-009: OmniInfer MCP Tool

**Priority**: Must
**Description**: The system SHALL expose an omni_infer MCP tool that supports forward, backward, bidirectional, and abductive inference on the knowledge graph.

```rust
/// MCP Tool: Perform omnidirectional inference
///
/// Supports forward, backward, bidirectional, and abductive inference
#[mcp_tool]
pub struct OmniInfer {
    /// Query node IDs
    pub query_nodes: Vec<String>,
    /// Inference direction
    pub direction: String,  // "forward", "backward", "bidirectional", "abduction"
    /// Clamped variables (node_id -> value)
    pub clamped: Option<HashMap<String, f32>>,
    /// Maximum depth
    pub max_depth: Option<u32>,
}

#[mcp_response]
pub struct OmniInferResponse {
    pub inferred_nodes: Vec<String>,
    pub beliefs: HashMap<String, f32>,
    pub direction: String,
    pub explanation: String,
}

/// Valid inference directions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceDirection {
    Forward,
    Backward,
    Bidirectional,
    Abduction,
}

impl TryFrom<&str> for InferenceDirection {
    type Error = ValidationError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "forward" => Ok(Self::Forward),
            "backward" => Ok(Self::Backward),
            "bidirectional" => Ok(Self::Bidirectional),
            "abduction" => Ok(Self::Abduction),
            _ => Err(ValidationError::InvalidEnumValue {
                field: "direction".to_string(),
                valid_values: vec![
                    "forward".to_string(),
                    "backward".to_string(),
                    "bidirectional".to_string(),
                    "abduction".to_string(),
                ],
                actual: s.to_string(),
            }),
        }
    }
}

impl OmniInfer {
    /// Maximum allowed query nodes
    const MAX_QUERY_NODES: usize = 100;
    /// Maximum allowed depth
    const MAX_DEPTH: u32 = 10;
    /// Maximum clamped variables
    const MAX_CLAMPED: usize = 1000;

    /// Execute omnidirectional inference
    pub async fn execute(
        &self,
        inference_engine: &InferenceEngine,
    ) -> Result<OmniInferResponse, McpHardeningError> {
        // Validate query nodes count
        if self.query_nodes.is_empty() {
            return Err(McpHardeningError::Validation(
                ValidationError::FieldRequired {
                    field: "query_nodes".to_string(),
                }
            ));
        }
        if self.query_nodes.len() > Self::MAX_QUERY_NODES {
            return Err(McpHardeningError::Validation(
                ValidationError::ArrayTooLong {
                    field: "query_nodes".to_string(),
                    max_elements: Self::MAX_QUERY_NODES,
                    actual_elements: self.query_nodes.len(),
                }
            ));
        }

        // Validate direction
        let direction = InferenceDirection::try_from(self.direction.as_str())?;

        // Validate depth
        let max_depth = self.max_depth.unwrap_or(5);
        if max_depth > Self::MAX_DEPTH {
            return Err(McpHardeningError::Validation(
                ValidationError::ValueOutOfRange {
                    field: "max_depth".to_string(),
                    min: 1,
                    max: Self::MAX_DEPTH as i64,
                    actual: max_depth as i64,
                }
            ));
        }

        // Validate clamped variables
        if let Some(ref clamped) = self.clamped {
            if clamped.len() > Self::MAX_CLAMPED {
                return Err(McpHardeningError::Validation(
                    ValidationError::ObjectTooLarge {
                        field: "clamped".to_string(),
                        max_properties: Self::MAX_CLAMPED,
                        actual_properties: clamped.len(),
                    }
                ));
            }
        }

        // Execute inference
        let result = inference_engine.infer(
            &self.query_nodes,
            direction.clone(),
            self.clamped.clone(),
            max_depth,
        ).await?;

        Ok(OmniInferResponse {
            inferred_nodes: result.inferred_nodes,
            beliefs: result.beliefs,
            direction: self.direction.clone(),
            explanation: result.generate_explanation(),
        })
    }
}
```

**Acceptance Criteria**:
- [ ] All four inference directions supported
- [ ] Query node count limited (max 100)
- [ ] Depth limited (max 10)
- [ ] Clamped variables validated
- [ ] Response includes belief values for inferred nodes
- [ ] Explanation generated for inference path

---

#### REQ-MCPH-010: VerifyCodeNode MCP Tool

**Priority**: Should
**Description**: The system SHOULD expose a verify_code_node MCP tool that uses Lean-inspired SMT verification from Module 6 for formal verification of code nodes.

```rust
/// MCP Tool: Verify code node with formal verification
///
/// Uses Lean-inspired SMT verification from Module 6
#[mcp_tool]
pub struct VerifyCodeNode {
    /// Node ID containing code
    pub node_id: String,
    /// Verification timeout in ms
    pub timeout_ms: Option<u64>,
    /// Specific conditions to check
    pub conditions: Option<Vec<String>>,
}

#[mcp_response]
pub struct VerifyCodeResponse {
    pub status: String,  // "verified", "failed", "timeout", "not_applicable"
    pub counterexample: Option<String>,
    pub conditions_checked: u32,
    pub conditions_passed: u32,
}

/// Verification status enumeration
#[derive(Debug, Clone, Serialize)]
pub enum VerificationStatus {
    Verified,
    Failed,
    Timeout,
    NotApplicable,
}

impl ToString for VerificationStatus {
    fn to_string(&self) -> String {
        match self {
            Self::Verified => "verified".to_string(),
            Self::Failed => "failed".to_string(),
            Self::Timeout => "timeout".to_string(),
            Self::NotApplicable => "not_applicable".to_string(),
        }
    }
}

impl VerifyCodeNode {
    /// Default verification timeout (5 seconds)
    const DEFAULT_TIMEOUT_MS: u64 = 5000;
    /// Maximum verification timeout (30 seconds)
    const MAX_TIMEOUT_MS: u64 = 30000;
    /// Maximum conditions to check
    const MAX_CONDITIONS: usize = 50;

    /// Execute formal verification on code node
    pub async fn execute(
        &self,
        verifier: &FormalVerifier,
        graph: &ContextGraph,
    ) -> Result<VerifyCodeResponse, McpHardeningError> {
        // Validate node ID format
        if !is_valid_node_id(&self.node_id) {
            return Err(McpHardeningError::Validation(
                ValidationError::InvalidFormat {
                    field: "node_id".to_string(),
                    expected: "Valid UUID".to_string(),
                    actual: self.node_id.clone(),
                }
            ));
        }

        // Validate timeout
        let timeout = self.timeout_ms.unwrap_or(Self::DEFAULT_TIMEOUT_MS);
        if timeout > Self::MAX_TIMEOUT_MS {
            return Err(McpHardeningError::Validation(
                ValidationError::ValueOutOfRange {
                    field: "timeout_ms".to_string(),
                    min: 1,
                    max: Self::MAX_TIMEOUT_MS as i64,
                    actual: timeout as i64,
                }
            ));
        }

        // Validate conditions count
        if let Some(ref conditions) = self.conditions {
            if conditions.len() > Self::MAX_CONDITIONS {
                return Err(McpHardeningError::Validation(
                    ValidationError::ArrayTooLong {
                        field: "conditions".to_string(),
                        max_elements: Self::MAX_CONDITIONS,
                        actual_elements: conditions.len(),
                    }
                ));
            }
        }

        // Fetch node from graph
        let node = graph.get_node(&self.node_id).await
            .map_err(|_| McpHardeningError::Internal(
                format!("Node not found: {}", self.node_id)
            ))?;

        // Check if node contains code
        if !node.has_code_content() {
            return Ok(VerifyCodeResponse {
                status: VerificationStatus::NotApplicable.to_string(),
                counterexample: None,
                conditions_checked: 0,
                conditions_passed: 0,
            });
        }

        // Execute verification with timeout
        let result = tokio::time::timeout(
            Duration::from_millis(timeout),
            verifier.verify(&node, self.conditions.clone()),
        ).await;

        match result {
            Ok(Ok(verification_result)) => Ok(VerifyCodeResponse {
                status: verification_result.status.to_string(),
                counterexample: verification_result.counterexample,
                conditions_checked: verification_result.conditions_checked,
                conditions_passed: verification_result.conditions_passed,
            }),
            Ok(Err(e)) => Ok(VerifyCodeResponse {
                status: VerificationStatus::Failed.to_string(),
                counterexample: Some(e.to_string()),
                conditions_checked: 0,
                conditions_passed: 0,
            }),
            Err(_) => Ok(VerifyCodeResponse {
                status: VerificationStatus::Timeout.to_string(),
                counterexample: None,
                conditions_checked: 0,
                conditions_passed: 0,
            }),
        }
    }
}

/// Validate node ID format (UUID)
fn is_valid_node_id(id: &str) -> bool {
    uuid::Uuid::parse_str(id).is_ok()
}
```

**Acceptance Criteria**:
- [ ] Formal verification accessible via MCP
- [ ] Node ID validated (UUID format)
- [ ] Timeout configurable (max 30s)
- [ ] Conditions count limited (max 50)
- [ ] Graceful timeout handling
- [ ] Counterexamples returned on verification failure
- [ ] Not-applicable status for non-code nodes

---

## 4. Error Handling

### 4.1 Sanitized Error Types

```rust
/// Production-safe error response.
///
/// All errors exposed to clients are sanitized to prevent information leakage.
/// Internal errors are logged with full details but clients receive generic messages.
#[derive(Debug, Error)]
pub enum McpHardeningError {
    /// Validation errors (safe to expose details)
    #[error("{0}")]
    Validation(#[from] ValidationError),

    /// Rate limit errors (safe to expose)
    #[error("{0}")]
    RateLimit(#[from] RateLimitError),

    /// Authentication errors (limited details)
    #[error("Authentication failed")]
    Authentication(#[source] AuthError),

    /// Quota errors (safe to expose)
    #[error("{0}")]
    Quota(#[from] QuotaError),

    /// Internal errors (never expose details)
    #[error("Internal server error")]
    Internal(String),
}

impl McpHardeningError {
    /// Get MCP error code
    pub fn error_code(&self) -> i32 {
        match self {
            Self::Validation(e) => e.error_code(),
            Self::RateLimit(e) => e.error_code(),
            Self::Authentication(e) => e.error_code(),
            Self::Quota(e) => e.error_code(),
            Self::Internal(_) => -32603,
        }
    }

    /// Get client-safe message
    pub fn client_message(&self) -> String {
        match self {
            Self::Validation(e) => e.to_string(),
            Self::RateLimit(e) => e.to_string(),
            Self::Authentication(_) => "Authentication failed".to_string(),
            Self::Quota(e) => e.to_string(),
            Self::Internal(_) => "Internal server error".to_string(),
        }
    }

    /// Log internal details
    pub fn log_details(&self) {
        match self {
            Self::Internal(details) => {
                tracing::error!(details = %details, "Internal error occurred");
            }
            Self::Authentication(e) => {
                tracing::warn!(error = %e, "Authentication failure");
            }
            _ => {}
        }
    }
}
```

**Acceptance Criteria**:
- [ ] No stack traces in client responses
- [ ] No file paths in error messages
- [ ] No internal function names exposed
- [ ] Internal details logged server-side
- [ ] Generic messages for sensitive errors

---

## 5. Performance Requirements

### 5.1 Latency Budgets

| Operation | Budget | Notes |
|-----------|--------|-------|
| Input Validation | <1ms | Schema validation + injection detection |
| Rate Limit Check | <100us | Token bucket evaluation |
| Auth Verification | <500us | Token lookup + permission check |
| Quota Check | <200us | Resource usage lookup |
| Audit Log Write | <200us | Async non-blocking write |
| Error Sanitization | <50us | Message cleaning |
| Total Middleware Overhead | <2ms | Combined security layer |

### 5.2 Throughput Requirements

| Metric | Target |
|--------|--------|
| Global Request Rate | 10,000 req/min |
| Per-Session Rate | 100 req/min |
| Concurrent Sessions | >1,000 |
| Audit Writes/sec | >10,000 |
| Authentication Verifications/sec | >5,000 |

### 5.3 Resource Constraints

| Resource | Limit |
|----------|-------|
| Memory per Session | 100MB |
| CPU Time per Request | 5ms |
| Storage per Session | 1GB |
| Audit Log Retention | 90 days |
| Session Token TTL | 24 hours |
| Rate Limit Window | 60 seconds |

---

## 6. Dependencies

### 6.1 Internal Module Dependencies

| Dependency | Purpose | Interface |
|------------|---------|-----------|
| Module 1: Ghost System | Trait definitions | Core traits |
| Module 2: Core Infrastructure | Storage, MCP base | RocksDB, MCP types |
| Module 11: Immune System | Threat detection | AdversarialDetector |
| Module 12: Active Inference | Session state | SessionBeliefs |
| Module 12.5: Steering Subsystem | Steering feedback | SteeringSubsystem |
| Module 6: Formal Verification | Code verification | FormalVerifier |
| Module 8: Inference Engine | Omnidirectional inference | InferenceEngine |

### 6.2 External Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| tower | 0.4+ | Middleware framework |
| tokio | 1.35+ | Async runtime |
| ring | 0.17+ | Cryptographic operations |
| sha2 | 0.10+ | Hash functions |
| regex | 1.10+ | Pattern matching |
| serde | 1.0+ | Serialization |
| serde_json | 1.0+ | JSON handling |
| thiserror | 1.0+ | Error handling |
| tracing | 0.1+ | Logging |
| base64 | 0.21+ | Token encoding |
| hex | 0.4+ | Hash encoding |
| uuid | 1.6+ | Identifiers |
| async-trait | 0.1+ | Async traits |

---

## 7. Traceability Matrix

### 7.1 Requirements to PRD

| Requirement | PRD Section | Description |
|-------------|-------------|-------------|
| REQ-MCPH-001 | 5.1 (Protocol) | Input validation for MCP |
| REQ-MCPH-002 | 10.1 (Adversarial Defense) | Injection detection |
| REQ-MCPH-003 | (Rate Limits) | Token bucket rate limiting |
| REQ-MCPH-004 | 5.5 (Meta-Cognitive) | Session authentication |
| REQ-MCPH-005 | 5.6 (Diagnostic) | Audit logging |
| REQ-MCPH-006 | 7.3 (Homeostatic) | Resource quotas |
| REQ-MCPH-007 | 5.1 (Protocol) | Middleware integration |
| REQ-MCPH-008 | 5.5 (Meta-Cognitive) | Marblestone steering access |
| REQ-MCPH-009 | 5.4 (Inference) | Omnidirectional inference |
| REQ-MCPH-010 | 5.3 (Verification) | Formal verification access |

### 7.2 Requirements to Implementation Plan

| Requirement | Implementation Plan Section |
|-------------|----------------------------|
| REQ-MCPH-001 | Module 13: Security Measures |
| REQ-MCPH-002 | Module 13: Security Measures |
| REQ-MCPH-003 | Module 13: Rate Limiting |
| REQ-MCPH-004 | Module 13: Authentication & Authorization |
| REQ-MCPH-005 | Module 13: Monitoring & Observability |
| REQ-MCPH-006 | Module 13: Resource Management |
| REQ-MCPH-007 | Module 13: Error Handling |
| REQ-MCPH-008 | Module 12.5: Steering Subsystem |
| REQ-MCPH-009 | Module 8: Inference Engine |
| REQ-MCPH-010 | Module 6: Formal Verification |

### 7.3 Requirements to Security Requirements

| Requirement | Security ID | Description |
|-------------|------------|-------------|
| REQ-MCPH-001 | SEC-02 | Input sanitization for all MCP endpoints |
| REQ-MCPH-002 | SEC-02 | Injection prevention |
| REQ-MCPH-003 | (Rate Limits) | 1000 req/min, 100 req/session |
| REQ-MCPH-004 | SEC-04 | Memory isolation between sessions |
| REQ-MCPH-005 | SEC-01 | Audit trail for compliance |
| REQ-MCPH-006 | SEC-04 | Resource isolation |
| REQ-MCPH-007 | SEC-02 | Defense in depth |
| REQ-MCPH-008 | SEC-02 | Input validation for steering |
| REQ-MCPH-009 | SEC-02 | Input validation for inference |
| REQ-MCPH-010 | SEC-02 | Input validation for verification |

---

## 8. Verification Methods

### 8.1 Unit Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| test_input_validation_schema | Schema validation accuracy | All valid inputs pass, invalid rejected |
| test_injection_detection | Injection pattern detection | >99% detection rate |
| test_rate_limit_token_bucket | Token bucket algorithm | Correct refill and consumption |
| test_auth_token_generation | Token security | Cryptographically random |
| test_auth_permission_levels | Permission enforcement | Correct access control |
| test_audit_chain_integrity | Tamper-evident logging | Chain verification passes |
| test_quota_enforcement | Resource limits | Violations blocked |
| test_error_sanitization | Error message safety | No information leakage |
| test_steering_feedback_validation | Steering feedback input validation | Content limits enforced |
| test_omni_infer_directions | OmniInfer direction validation | All 4 directions work |
| test_omni_infer_limits | OmniInfer query limits | Node and depth limits enforced |
| test_verify_code_timeout | Verification timeout handling | Graceful timeout response |
| test_verify_code_non_code_node | Non-code node verification | Returns not_applicable |

### 8.2 Integration Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| test_middleware_pipeline | Full request flow | All layers execute correctly |
| test_concurrent_requests | Load handling | No race conditions |
| test_session_lifecycle | Session management | Create, verify, refresh, revoke |
| test_audit_query | Log querying | Correct results returned |
| test_graceful_degradation | Overload handling | System remains available |
| test_steering_with_module_12 | Steering subsystem integration | Scores returned correctly |
| test_omni_infer_graph | Inference engine integration | Beliefs propagated correctly |
| test_verify_with_module_6 | Formal verifier integration | Verification results correct |

### 8.3 Security Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| test_prompt_injection | Prompt injection attacks | All blocked |
| test_sql_injection | SQL injection attacks | All blocked |
| test_path_traversal | Path traversal attacks | All blocked |
| test_dos_resistance | DoS attack simulation | 99.9% uptime |
| test_brute_force | Auth brute force | Lockout triggered |
| test_timing_attacks | Timing side channels | Constant-time operations |

### 8.4 Performance Benchmarks

| Benchmark | Description | Target |
|-----------|-------------|--------|
| bench_validation | Input validation throughput | >100K/sec |
| bench_rate_limit | Rate limit check latency | <100us P99 |
| bench_auth_verify | Token verification | <500us P99 |
| bench_audit_write | Audit log throughput | >10K/sec |
| bench_middleware | Full middleware latency | <2ms P99 |
| bench_steering_feedback | Steering feedback latency | <5ms P99 |
| bench_omni_infer | Omnidirectional inference | <10ms P99 (depth 5) |
| bench_verify_code | Formal verification | <5s P99 (simple code) |

---

## 9. Quality Gates

| Gate | Criteria | Verification |
|------|----------|--------------|
| Unit Test Coverage | >90% | cargo tarpaulin |
| Security Tests | All pass | cargo test --test security |
| Performance Targets | All met | cargo bench |
| OWASP Top 10 | All addressed | Security checklist |
| Penetration Test | No critical findings | External audit |
| Error Leakage | Zero findings | Security review |
| Documentation | All public APIs | cargo doc |
| Code Review | Security-focused | Pull request review |

---

## 10. Deployment Considerations

### 10.1 Configuration Recommendations

```yaml
# Production configuration
mcp_hardening:
  input_validation:
    max_request_size: 1048576  # 1MB
    max_string_length: 65536   # 64KB
    strict_mode: true

  rate_limiting:
    global_rpm: 10000
    session_rpm: 100
    burst_multiplier: 2.0

  authentication:
    session_ttl: 86400         # 24 hours
    refresh_ttl: 604800        # 7 days
    inactivity_timeout: 3600   # 1 hour

  audit:
    retention_days: 90
    enable_chaining: true
    mask_pii: true

  quotas:
    max_memory_per_session: 104857600  # 100MB
    max_storage_per_session: 1073741824  # 1GB
    max_concurrent_requests: 10
```

### 10.2 Monitoring Alerts

| Alert | Condition | Action |
|-------|-----------|--------|
| High Rejection Rate | >5% requests rejected | Investigate patterns |
| Rate Limit Spikes | >10 sessions blocked/min | Check for attacks |
| Auth Failures | >100 failures/min | Enable lockout |
| Quota Violations | >50/min | Review limits |
| Audit Chain Failure | Any failure | Investigate tampering |

---

*Document Version: 1.0.0*
*Generated: 2025-12-31*
*Based on PRD v2.0.0 and Implementation Plan Module 13*
