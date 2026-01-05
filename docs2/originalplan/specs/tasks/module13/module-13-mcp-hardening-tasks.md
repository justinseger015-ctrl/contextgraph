# Module 13: MCP Hardening - Atomic Tasks

```yaml
metadata:
  module_id: "module-13"
  module_name: "MCP Hardening"
  version: "1.0.0"
  phase: 12
  total_tasks: 24
  approach: "defense-in-depth"
  created: "2025-12-31"
  dependencies:
    - module-01-ghost-system
    - module-02-core-infrastructure
    - module-06-bio-nervous-system
    - module-11-immune-system
    - module-12-active-inference
    - module-12.5-steering-subsystem
  estimated_duration: "4 weeks"
  spec_refs:
    - SPEC-MCPH-013 (Functional)
    - TECH-MCPH-013 (Technical)
```

---

## Task Overview

This module implements comprehensive security hardening for the Model Context Protocol (MCP) layer with defense-in-depth architecture. The system provides input validation with injection prevention, token bucket rate limiting, session-based authentication with permission levels, tamper-evident audit logging, resource quota management, and Tower-pattern security middleware composing all components.

### Key Components

| Component | Purpose | Performance Target |
|-----------|---------|-------------------|
| InputValidator | Schema validation, injection detection | <1ms per request |
| RateLimiter | Token bucket rate limiting | <100us check |
| AuthManager | Session tokens, permission levels | <500us verification |
| AuditLogger | Tamper-evident async logging | <200us write |
| ResourceQuotaManager | Memory/CPU/storage quotas | <100us check |
| SecurityMiddleware | Tower-pattern composition | Full pipeline <5ms |

### Marblestone MCP Tools

1. `get_steering_feedback` - Elevated (Level 2), steering guidance
2. `omni_infer` - Elevated (Level 2), omnidirectional inference with clamped values
3. `verify_code_node` - Admin (Level 3), code verification with signatures

### Task Organization

1. **Foundation Layer** (Tasks 1-8): Core types, schemas, patterns, configuration
2. **Rate Limiting Layer** (Tasks 9-12): Token bucket, scopes, quota info
3. **Authentication Layer** (Tasks 13-16): Sessions, permissions, tool authorization
4. **Audit Layer** (Tasks 17-19): Tamper-evident logging, chain hashing, queries
5. **Resource Layer** (Tasks 20-21): Quota management, RAII guards
6. **Integration Layer** (Tasks 22-24): SecurityMiddleware, MCP tools, pipeline

---

## Foundation Layer: Core Types & Validation

```yaml
tasks:
  # ============================================================
  # FOUNDATION: Input Validation Types
  # ============================================================

  - id: "TASK-13-001"
    title: "Define ParameterSchema and ParamType Enumerations"
    description: |
      Implement core type definitions for MCP parameter validation schemas.

      Structs to implement:
      - ParameterSchema: name, param_type, required, constraints, sanitizers
      - ParamType enum: String{min_len, max_len}, Integer{min, max}, Float{min, max},
        Boolean, Uuid, Array{item_type, max_items}, Object{schema}
      - Constraint enum: Regex, AllowedValues, NoSqlInjection, NoXss,
        NoCommandInjection, NoPathTraversal, Custom
      - SanitizerType enum: HtmlEscape, SqlEscape, ShellEscape, PathNormalize,
        UnicodeNormalize, Trim

      All types must derive Serialize, Deserialize, Debug, Clone.
      Use serde for JSON schema compatibility.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-mcp-hardening/src/validation/schema.rs"
    requirements_traced:
      - REQ-MCPH-001
      - REQ-MCPH-002
    dependencies: []
    input_dependencies: []
    output_deliverables:
      - "ParameterSchema struct with all fields"
      - "ParamType enum with nested type support"
      - "Constraint enum with 7 variants"
      - "SanitizerType enum with 6 variants"
    acceptance_criteria:
      - "All types compile with serde derives"
      - "ParamType supports recursive Array and Object types"
      - "Constraint::Regex variant holds valid regex pattern string"
      - "Unit tests validate serialization roundtrip for all types"
      - "Documentation includes examples for each ParamType variant"

  - id: "TASK-13-002"
    title: "Implement ValidationResult and Error Types"
    description: |
      Create comprehensive validation result types with threat detection support.

      Types to implement:
      - ValidationResult: valid (bool), errors (Vec<ValidationError>),
        sanitized_input (Option<Value>), threat_indicators (Vec<ThreatIndicator>)
      - ValidationError: field (String), message (String), code (ValidationErrorCode)
      - ValidationErrorCode enum: MissingRequired, TypeMismatch, ConstraintViolation,
        InjectionDetected, SchemaMissing
      - ThreatIndicator: threat_type (ThreatType), confidence (f32), payload_snippet (String)
      - ThreatType enum: SqlInjection, XssAttempt, CommandInjection, PathTraversal,
        BufferOverflow, FormatString

      ThreatIndicator.confidence must be in range [0.0, 1.0].
      payload_snippet must be truncated to 100 chars max (no full payload leakage).
    layer: "foundation"
    priority: "critical"
    estimated_hours: 2
    file_path: "crates/context-graph-mcp-hardening/src/validation/result.rs"
    requirements_traced:
      - REQ-MCPH-001
      - REQ-MCPH-002
    dependencies:
      - TASK-13-001
    input_dependencies:
      - "ParameterSchema from TASK-13-001"
    output_deliverables:
      - "ValidationResult struct"
      - "ValidationError with error codes"
      - "ThreatIndicator with confidence scoring"
      - "ThreatType enum covering OWASP Top 10 injection types"
    acceptance_criteria:
      - "ValidationResult.valid is false when errors non-empty"
      - "ThreatIndicator.confidence clamped to [0.0, 1.0]"
      - "payload_snippet truncated at 100 characters"
      - "All error codes map to distinct failure modes"
      - "Unit tests cover all ValidationErrorCode variants"

  - id: "TASK-13-003"
    title: "Compile Injection Detection Regex Patterns"
    description: |
      Pre-compile all injection detection patterns at initialization time for
      sub-millisecond validation performance.

      Pattern sets from technical spec:
      - SQL_INJECTION_PATTERNS (7 patterns): union, select, insert, comment markers,
        boolean conditions, hex encoding, char/concat functions
      - XSS_PATTERNS (6 patterns): script tags, javascript:, event handlers,
        iframe, expression(), data:text/html
      - COMMAND_INJECTION_PATTERNS (6 patterns): shell metacharacters, $(),
        backticks, eval/exec, pipes, redirects
      - PATH_TRAVERSAL_PATTERNS (5 patterns): ../, URL encoded variants,
        double-encoded, null byte injection

      Use regex crate with RegexSet for parallel matching.
      All patterns must be case-insensitive where applicable.
    layer: "foundation"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-mcp-hardening/src/validation/patterns.rs"
    requirements_traced:
      - REQ-MCPH-002
    dependencies:
      - TASK-13-002
    input_dependencies:
      - "ThreatType enum from TASK-13-002"
    output_deliverables:
      - "CompiledPatternSet struct with RegexSet"
      - "Pattern compilation at startup (lazy_static or OnceCell)"
      - "detect_injection(input: &str) -> Vec<ThreatIndicator> function"
      - "Benchmark tests proving <500us for 10KB input"
    acceptance_criteria:
      - "All 24 patterns compile without regex errors"
      - "RegexSet enables parallel pattern matching"
      - "detect_injection returns ThreatIndicator for each match"
      - "Confidence score based on pattern specificity (0.7-1.0)"
      - "Performance: <500us for 10KB input string"
      - "Unit tests with known injection payloads achieve 100% detection"

  - id: "TASK-13-004"
    title: "Implement Content Sanitizers"
    description: |
      Create sanitization functions for each SanitizerType to neutralize
      dangerous content while preserving legitimate data.

      Sanitizers to implement:
      - HtmlEscape: Convert <>&"' to HTML entities
      - SqlEscape: Escape single quotes, backslashes
      - ShellEscape: Quote/escape shell metacharacters
      - PathNormalize: Resolve . and .., remove duplicate slashes
      - UnicodeNormalize: Apply NFC normalization, strip zero-width chars
      - Trim: Remove leading/trailing whitespace and control characters

      Each sanitizer must be idempotent (applying twice = applying once).
      Chain sanitizers in order specified by ParameterSchema.sanitizers.
    layer: "foundation"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-mcp-hardening/src/validation/sanitizer.rs"
    requirements_traced:
      - REQ-MCPH-001
      - REQ-MCPH-002
    dependencies:
      - TASK-13-001
    input_dependencies:
      - "SanitizerType enum from TASK-13-001"
    output_deliverables:
      - "Sanitizer trait with apply(&str) -> String method"
      - "Implementations for all 6 SanitizerType variants"
      - "SanitizerChain for composing multiple sanitizers"
      - "Idempotency tests for each sanitizer"
    acceptance_criteria:
      - "HtmlEscape converts all 5 characters to entities"
      - "SqlEscape escapes ' as '' and \\ as \\\\"
      - "ShellEscape produces shell-safe strings for /bin/sh"
      - "PathNormalize removes ../ sequences completely"
      - "UnicodeNormalize strips zero-width characters (U+200B, U+FEFF)"
      - "All sanitizers are idempotent (sanitize(sanitize(x)) == sanitize(x))"
      - "Performance: <100us per sanitizer for 1KB input"

  - id: "TASK-13-005"
    title: "Implement InputValidator Trait and Core Logic"
    description: |
      Create the main InputValidator implementation with async validation,
      schema registration, injection detection, and sanitization pipeline.

      Trait methods:
      - register_schema(tool_name: &str, schema: Vec<ParameterSchema>) -> Result<()>
      - validate(tool_name: &str, input: &Value) -> ValidationResult
      - detect_injection(input: &str) -> Vec<ThreatIndicator>
      - sanitize(tool_name: &str, input: Value) -> Result<Value>

      Implementation requirements:
      - Schema lookup by tool name with O(1) HashMap
      - Recursive validation for nested Object and Array types
      - Short-circuit on first critical threat detection
      - Collect all validation errors before returning
    layer: "foundation"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-mcp-hardening/src/validation/validator.rs"
    requirements_traced:
      - REQ-MCPH-001
      - REQ-MCPH-002
    dependencies:
      - TASK-13-001
      - TASK-13-002
      - TASK-13-003
      - TASK-13-004
    input_dependencies:
      - "ParameterSchema from TASK-13-001"
      - "ValidationResult from TASK-13-002"
      - "CompiledPatternSet from TASK-13-003"
      - "Sanitizers from TASK-13-004"
    output_deliverables:
      - "InputValidator trait definition"
      - "DefaultInputValidator implementation"
      - "Schema registry with HashMap<String, Vec<ParameterSchema>>"
      - "Validation metrics collection"
    acceptance_criteria:
      - "validate() returns ValidationResult in <1ms for typical requests"
      - "All registered schemas retrievable by tool name"
      - "Recursive validation handles 10 levels of nesting"
      - "Threat indicators include confidence and snippet"
      - "Sanitized output returned only when valid=true"
      - "Integration test: validate all 3 Marblestone tool schemas"

  - id: "TASK-13-006"
    title: "Define Size Limits and Configuration"
    description: |
      Create comprehensive configuration for input validation with sensible
      defaults matching technical specification.

      Configuration struct fields:
      - max_input_size_bytes: 1MB (1_048_576)
      - max_array_items: 1000
      - max_string_length: 65536 (64KB)
      - max_nesting_depth: 10
      - enable_injection_detection: true
      - injection_patterns: {sql: true, xss: true, command: true, path_traversal: true}

      Load from YAML configuration file with validation.
      Support runtime reconfiguration without restart.
    layer: "foundation"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-mcp-hardening/src/config/validation.rs"
    requirements_traced:
      - REQ-MCPH-001
    dependencies: []
    input_dependencies: []
    output_deliverables:
      - "ValidationConfig struct with defaults"
      - "YAML configuration schema"
      - "Runtime reconfiguration support"
      - "Validation of config values on load"
    acceptance_criteria:
      - "Default config matches technical spec values"
      - "YAML config loads without panic on valid input"
      - "Invalid config values return descriptive errors"
      - "Runtime config updates take effect within 1 second"
      - "Config validation rejects negative or zero limits"

  - id: "TASK-13-007"
    title: "Register Marblestone Tool Schemas"
    description: |
      Define and register validation schemas for all three Marblestone MCP tools:
      get_steering_feedback, omni_infer, and verify_code_node.

      Schemas:
      1. get_steering_feedback:
         - session_id: String (required), UUID format, NoSqlInjection

      2. omni_infer:
         - direction: String (required), 1-256 chars, NoCommandInjection
         - clamped: Array<ClampedValue> (required), max 100 items
           - ClampedValue: name (String), value (Float), min (Float), max (Float)

      3. verify_code_node:
         - node_id: Uuid (required)

      Include custom validation: ClampedValue.value must be within [min, max].
    layer: "foundation"
    priority: "high"
    estimated_hours: 3
    file_path: "crates/context-graph-mcp-hardening/src/validation/marblestone_schemas.rs"
    requirements_traced:
      - REQ-MCPH-008
      - REQ-MCPH-009
      - REQ-MCPH-010
    dependencies:
      - TASK-13-001
      - TASK-13-005
    input_dependencies:
      - "ParameterSchema from TASK-13-001"
      - "InputValidator from TASK-13-005"
    output_deliverables:
      - "get_steering_feedback_schema() function"
      - "omni_infer_schema() function"
      - "verify_code_node_schema() function"
      - "register_marblestone_schemas(validator: &mut InputValidator)"
    acceptance_criteria:
      - "All three schemas register successfully"
      - "session_id rejects non-UUID strings"
      - "direction rejects strings > 256 chars"
      - "clamped array rejects > 100 items"
      - "ClampedValue validation ensures value in [min, max]"
      - "Integration test validates sample valid/invalid inputs"

  - id: "TASK-13-008"
    title: "Implement ValidatorError Type and Error Handling"
    description: |
      Create comprehensive error type for validator failures with security-safe
      error messages that don't leak internal details.

      ValidatorError variants:
      - SchemaMissing(tool_name: String)
      - ValidationFailed(errors: Vec<ValidationError>)
      - InjectionDetected(indicators: Vec<ThreatIndicator>)
      - SizeLimitExceeded { field: String, size: usize, max: usize }
      - NestingDepthExceeded { depth: usize, max: usize }
      - ConfigError(message: String)

      Error messages must be safe to return to clients (no stack traces,
      no internal paths, no sensitive data).
    layer: "foundation"
    priority: "medium"
    estimated_hours: 2
    file_path: "crates/context-graph-mcp-hardening/src/validation/error.rs"
    requirements_traced:
      - REQ-MCPH-001
    dependencies:
      - TASK-13-002
    input_dependencies:
      - "ValidationError from TASK-13-002"
      - "ThreatIndicator from TASK-13-002"
    output_deliverables:
      - "ValidatorError enum with thiserror derives"
      - "Safe error message formatting"
      - "Error code mapping for client responses"
    acceptance_criteria:
      - "All variants implement Display with safe messages"
      - "No internal paths or stack traces in error messages"
      - "Error codes are numeric for programmatic handling"
      - "InjectionDetected hides full payload (only shows truncated snippet)"
      - "Unit tests verify error message safety"
```

---

## Rate Limiting Layer: Token Bucket Implementation

```yaml
  # ============================================================
  # RATE LIMITING: Token Bucket Algorithm
  # ============================================================

  - id: "TASK-13-009"
    title: "Implement TokenBucket Core Algorithm"
    description: |
      Create the core token bucket data structure with thread-safe refill logic.

      TokenBucket struct:
      - tokens: AtomicF64 or RwLock<f64> for current tokens
      - last_refill: AtomicU64 (timestamp in microseconds)
      - config: BucketConfig (capacity, refill_rate, burst_size)

      Methods:
      - new(config: BucketConfig) -> Self (start at full capacity)
      - try_consume(tokens: u64) -> bool (refill then consume)
      - refill() (add tokens based on elapsed time, cap at capacity)
      - time_until_available(tokens: u64) -> Duration
      - tokens_remaining() -> u64

      Must be thread-safe for concurrent access.
      Use compare-and-swap for lock-free refill where possible.
    layer: "rate_limiting"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-mcp-hardening/src/rate_limit/bucket.rs"
    requirements_traced:
      - REQ-MCPH-003
    dependencies: []
    input_dependencies: []
    output_deliverables:
      - "TokenBucket struct with atomic operations"
      - "BucketConfig struct with capacity, refill_rate, burst_size"
      - "Thread-safe try_consume and refill methods"
      - "Time calculation for retry-after header"
    acceptance_criteria:
      - "TokenBucket is Send + Sync"
      - "try_consume returns false when tokens insufficient"
      - "refill adds tokens proportional to elapsed time"
      - "tokens never exceed capacity after refill"
      - "time_until_available returns correct Duration"
      - "Concurrent access test with 100 threads passes"
      - "Performance: try_consume < 1us"

  - id: "TASK-13-010"
    title: "Define RateLimitScope and Composite Scopes"
    description: |
      Create scope definitions for hierarchical rate limiting with composite
      scope support for multi-dimensional limits.

      RateLimitScope enum:
      - Global
      - Session(session_id: String)
      - User(user_id: Uuid)
      - Tool(tool_name: String)
      - IpAddress(ip: IpAddr)
      - Composite(scopes: Vec<RateLimitScope>)

      Composite scope checks ALL constituent scopes and returns the most
      restrictive result. Example: Session + Tool scope.

      Implement Hash and Eq for use as HashMap keys.
    layer: "rate_limiting"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-mcp-hardening/src/rate_limit/scope.rs"
    requirements_traced:
      - REQ-MCPH-003
    dependencies: []
    input_dependencies: []
    output_deliverables:
      - "RateLimitScope enum with 6 variants"
      - "Hash and Eq implementations"
      - "Composite scope flattening logic"
      - "Display implementation for logging"
    acceptance_criteria:
      - "All scope variants hashable and comparable"
      - "Composite flattens nested Composite variants"
      - "Display shows human-readable scope description"
      - "Same scope values produce same hash"
      - "Unit tests verify hash consistency"

  - id: "TASK-13-011"
    title: "Implement RateLimiter Trait and Default Configuration"
    description: |
      Create the RateLimiter trait and default implementation with bucket
      management per scope.

      Trait methods:
      - check(scope: &RateLimitScope) -> RateLimitDecision
      - acquire(scope: &RateLimitScope, tokens: u64) -> RateLimitDecision
      - configure(scope: RateLimitScope, config: BucketConfig)
      - reset(scope: &RateLimitScope)
      - quota(scope: &RateLimitScope) -> QuotaInfo

      RateLimitDecision struct:
      - allowed: bool
      - tokens_remaining: u64
      - retry_after: Option<Duration>
      - scope: RateLimitScope
      - quota_info: QuotaInfo

      Default configurations from spec:
      - Global: 1000/min, burst 100
      - Session: 100/min, burst 20
      - omni_infer tool: 50/min, burst 10
    layer: "rate_limiting"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-mcp-hardening/src/rate_limit/limiter.rs"
    requirements_traced:
      - REQ-MCPH-003
    dependencies:
      - TASK-13-009
      - TASK-13-010
    input_dependencies:
      - "TokenBucket from TASK-13-009"
      - "RateLimitScope from TASK-13-010"
    output_deliverables:
      - "RateLimiter trait definition"
      - "DefaultRateLimiter implementation"
      - "RateLimitDecision and QuotaInfo structs"
      - "Default configuration functions"
    acceptance_criteria:
      - "check() returns decision without consuming tokens"
      - "acquire() consumes tokens and returns decision"
      - "Composite scope returns most restrictive result"
      - "retry_after calculated correctly when denied"
      - "Default configs match spec values exactly"
      - "Integration test: burst allows 100 rapid requests then blocks"

  - id: "TASK-13-012"
    title: "Implement Rate Limit Metrics and Monitoring"
    description: |
      Add metrics collection for rate limiting to track usage patterns,
      blocked requests, and bucket utilization.

      Metrics to collect:
      - requests_total: Counter by scope
      - requests_allowed: Counter by scope
      - requests_blocked: Counter by scope
      - bucket_utilization: Gauge (tokens_remaining / capacity) by scope
      - retry_after_seconds: Histogram

      Export as Prometheus metrics.
      Include per-scope breakdown and global aggregates.
    layer: "rate_limiting"
    priority: "medium"
    estimated_hours: 3
    file_path: "crates/context-graph-mcp-hardening/src/rate_limit/metrics.rs"
    requirements_traced:
      - REQ-MCPH-003
    dependencies:
      - TASK-13-011
    input_dependencies:
      - "RateLimiter from TASK-13-011"
      - "RateLimitDecision from TASK-13-011"
    output_deliverables:
      - "RateLimitMetrics struct"
      - "Prometheus metric registration"
      - "Per-scope metric labels"
      - "Metrics recording hooks in RateLimiter"
    acceptance_criteria:
      - "All 5 metric types implemented"
      - "Prometheus /metrics endpoint returns valid output"
      - "Scope labels correctly identify rate limit scope"
      - "Histogram buckets: [0.001, 0.01, 0.1, 1, 5, 10, 30, 60]"
      - "Metrics update atomically on each request"
```

---

## Authentication Layer: Sessions and Permissions

```yaml
  # ============================================================
  # AUTHENTICATION: Session Management
  # ============================================================

  - id: "TASK-13-013"
    title: "Define SessionToken and PermissionLevel Structures"
    description: |
      Create session token structure with embedded claims and permission levels.

      SessionToken struct:
      - session_id: Uuid
      - user_id: Option<Uuid>
      - permission_level: PermissionLevel
      - granted_tools: HashSet<String>
      - denied_tools: HashSet<String>
      - issued_at: DateTime<Utc>
      - expires_at: DateTime<Utc>
      - refresh_token: Option<String>
      - metadata: SessionMetadata

      PermissionLevel enum (ordered for comparison):
      - Public = 0 (read-only, no auth)
      - Authenticated = 1 (basic tool access)
      - Elevated = 2 (steering, inference)
      - Admin = 3 (verification, system)
      - System = 4 (internal only)

      SessionMetadata: client_ip, user_agent, device_fingerprint, mfa_verified, last_activity
    layer: "authentication"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-mcp-hardening/src/auth/session.rs"
    requirements_traced:
      - REQ-MCPH-004
      - REQ-MCPH-005
    dependencies: []
    input_dependencies: []
    output_deliverables:
      - "SessionToken struct with all fields"
      - "PermissionLevel enum with Ord trait"
      - "SessionMetadata struct"
      - "Serialization/deserialization support"
    acceptance_criteria:
      - "PermissionLevel comparison: Public < Authenticated < Elevated < Admin < System"
      - "SessionToken serializes to JWT-compatible format"
      - "expires_at validation: reject tokens past expiry"
      - "granted_tools overrides denied_tools for specific access"
      - "Unit tests cover permission level ordering"

  - id: "TASK-13-014"
    title: "Define ToolPermission Registry"
    description: |
      Create tool permission requirements for authorization checks.

      ToolPermission struct:
      - tool_name: String
      - required_level: PermissionLevel
      - additional_claims: Vec<String> (e.g., "steering:read")
      - rate_limit_override: Option<BucketConfig>

      Register Marblestone tool permissions:
      - get_steering_feedback: Elevated, ["steering:read"]
      - omni_infer: Elevated, ["inference:execute"], 50/min rate limit
      - verify_code_node: Admin, ["verification:execute"]

      Function: marblestone_tool_permissions() -> Vec<ToolPermission>
    layer: "authentication"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-mcp-hardening/src/auth/permissions.rs"
    requirements_traced:
      - REQ-MCPH-005
      - REQ-MCPH-008
      - REQ-MCPH-009
      - REQ-MCPH-010
    dependencies:
      - TASK-13-013
      - TASK-13-009
    input_dependencies:
      - "PermissionLevel from TASK-13-013"
      - "BucketConfig from TASK-13-009"
    output_deliverables:
      - "ToolPermission struct"
      - "marblestone_tool_permissions() function"
      - "Permission registry lookup by tool name"
    acceptance_criteria:
      - "All 3 Marblestone tools have registered permissions"
      - "omni_infer has rate limit override of 50/min"
      - "verify_code_node requires Admin level"
      - "Lookup by tool name returns Option<ToolPermission>"
      - "Unknown tools return None"

  - id: "TASK-13-015"
    title: "Implement AuthManager Trait and Session Lifecycle"
    description: |
      Create AuthManager trait for session creation, validation, and authorization.

      Trait methods:
      - create_session(user_id, permission_level, ttl, metadata) -> Result<SessionToken>
      - validate_session(token: &str) -> Result<SessionToken>
      - authorize_tool(session: &SessionToken, tool_name: &str) -> AuthDecision
      - refresh_session(refresh_token: &str) -> Result<SessionToken>
      - revoke_session(session_id: Uuid) -> Result<()>
      - register_tool_permission(permission: ToolPermission)
      - elevate_session(session_id, new_level, mfa_token) -> Result<SessionToken>

      AuthDecision struct:
      - allowed: bool
      - reason: Option<String>
      - session: Option<SessionToken>
      - required_level: PermissionLevel
      - actual_level: PermissionLevel

      Session storage: in-memory HashMap with RwLock (production: Redis).
    layer: "authentication"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-mcp-hardening/src/auth/manager.rs"
    requirements_traced:
      - REQ-MCPH-004
      - REQ-MCPH-005
    dependencies:
      - TASK-13-013
      - TASK-13-014
    input_dependencies:
      - "SessionToken from TASK-13-013"
      - "ToolPermission from TASK-13-014"
    output_deliverables:
      - "AuthManager trait definition"
      - "DefaultAuthManager implementation"
      - "AuthDecision struct"
      - "Session storage with expiry cleanup"
    acceptance_criteria:
      - "create_session generates unique session_id"
      - "validate_session rejects expired tokens"
      - "authorize_tool checks permission level and claims"
      - "refresh_session extends session with new token"
      - "revoke_session removes session from storage"
      - "elevate_session requires mfa_verified=true"
      - "Performance: validate_session < 500us"

  - id: "TASK-13-016"
    title: "Implement AuthError Type and Security Responses"
    description: |
      Create authentication error types with security-safe responses that
      prevent user enumeration and timing attacks.

      AuthError variants:
      - InvalidToken (generic, no detail on why)
      - SessionExpired
      - SessionRevoked
      - InsufficientPermission { required: PermissionLevel, actual: PermissionLevel }
      - MfaRequired
      - RateLimited { retry_after: Duration }
      - Internal(message: String)

      Security requirements:
      - InvalidToken returns same error for non-existent and malformed tokens
      - No timing differences between valid/invalid token checks
      - Internal details logged but not returned to client
    layer: "authentication"
    priority: "high"
    estimated_hours: 2
    file_path: "crates/context-graph-mcp-hardening/src/auth/error.rs"
    requirements_traced:
      - REQ-MCPH-004
    dependencies:
      - TASK-13-013
    input_dependencies:
      - "PermissionLevel from TASK-13-013"
    output_deliverables:
      - "AuthError enum with thiserror"
      - "Constant-time token comparison utility"
      - "Safe error messages for clients"
    acceptance_criteria:
      - "InvalidToken message is generic: 'authentication failed'"
      - "Token comparison uses constant-time comparison"
      - "No stack traces in error responses"
      - "Internal errors logged at error level"
      - "Unit test verifies constant-time comparison"
```

---

## Audit Layer: Tamper-Evident Logging

```yaml
  # ============================================================
  # AUDIT: Tamper-Evident Logging
  # ============================================================

  - id: "TASK-13-017"
    title: "Define AuditEvent and Chain Hash Structures"
    description: |
      Create audit event structure with tamper-evident chain hashing.

      AuditEvent struct:
      - event_id: Uuid
      - sequence_number: u64 (monotonically increasing)
      - timestamp: DateTime<Utc>
      - event_type: AuditEventType
      - session_id: Option<Uuid>
      - user_id: Option<Uuid>
      - tool_name: Option<String>
      - action: String
      - outcome: AuditOutcome
      - metadata: serde_json::Value
      - chain_hash: String (SHA-256 of previous event)

      AuditEventType enum:
      - Authentication, Authorization, ToolExecution, RateLimitHit,
        ValidationFailure, SecurityAlert, ConfigChange, SystemEvent

      AuditOutcome enum:
      - Success, Failure{reason}, Blocked{reason}, Throttled

      Chain hash: SHA256(prev_hash || seq_num || event_id || timestamp || action)
    layer: "audit"
    priority: "critical"
    estimated_hours: 3
    file_path: "crates/context-graph-mcp-hardening/src/audit/event.rs"
    requirements_traced:
      - REQ-MCPH-006
    dependencies: []
    input_dependencies: []
    output_deliverables:
      - "AuditEvent struct with all fields"
      - "AuditEventType enum with 8 variants"
      - "AuditOutcome enum with 4 variants"
      - "compute_chain_hash(prev_hash, event) -> String function"
    acceptance_criteria:
      - "sequence_number is globally unique and increasing"
      - "chain_hash links to previous event immutably"
      - "compute_chain_hash produces consistent output"
      - "AuditEvent serializes to JSON for storage"
      - "Unit test verifies chain integrity across 1000 events"

  - id: "TASK-13-018"
    title: "Implement AuditLogger Trait with Async Buffering"
    description: |
      Create async audit logger with bounded channel for sub-200us latency.

      Trait methods:
      - log(event: AuditEvent) -> Result<u64> (returns sequence number)
      - log_batch(events: Vec<AuditEvent>) -> Result<Vec<u64>>
      - query(query: AuditQuery) -> Result<Vec<TamperEvidentEntry>>
      - verify_integrity(from_seq: u64, to_seq: u64) -> Result<bool>
      - export(query: AuditQuery, format: ExportFormat) -> Result<Vec<u8>>
      - current_sequence() -> u64

      Implementation:
      - Bounded async channel (mpsc) with 10,000 capacity
      - Background writer task flushes every 100ms or 1000 events
      - Chain hash computed in writer thread
      - Sequence number via AtomicU64

      Performance target: log() completes in <200us.
    layer: "audit"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-mcp-hardening/src/audit/logger.rs"
    requirements_traced:
      - REQ-MCPH-006
    dependencies:
      - TASK-13-017
    input_dependencies:
      - "AuditEvent from TASK-13-017"
    output_deliverables:
      - "AuditLogger trait definition"
      - "AsyncAuditLogger implementation"
      - "Background writer task"
      - "TamperEvidentEntry with signature"
    acceptance_criteria:
      - "log() returns in <200us (P99)"
      - "Background flush interval is 100ms"
      - "Channel backpressure doesn't block caller beyond 200us"
      - "Sequence numbers are monotonically increasing"
      - "verify_integrity validates chain hashes"
      - "Benchmark: sustain 50,000 events/sec without dropping"

  - id: "TASK-13-019"
    title: "Implement Audit Query and Export"
    description: |
      Create audit query system for retrieving and exporting audit logs.

      AuditQuery struct:
      - start_time: Option<DateTime<Utc>>
      - end_time: Option<DateTime<Utc>>
      - event_types: Option<Vec<AuditEventType>>
      - session_id: Option<Uuid>
      - user_id: Option<Uuid>
      - tool_name: Option<String>
      - outcome: Option<AuditOutcome>
      - limit: usize (default 1000)
      - offset: usize (default 0)

      ExportFormat enum:
      - Json, Csv, Parquet

      Query execution should use indexes on timestamp, session_id, user_id.
      Export must include chain hashes for tamper verification.
    layer: "audit"
    priority: "high"
    estimated_hours: 4
    file_path: "crates/context-graph-mcp-hardening/src/audit/query.rs"
    requirements_traced:
      - REQ-MCPH-006
    dependencies:
      - TASK-13-017
      - TASK-13-018
    input_dependencies:
      - "AuditEvent from TASK-13-017"
      - "AuditLogger from TASK-13-018"
    output_deliverables:
      - "AuditQuery struct"
      - "Query execution with filtering"
      - "ExportFormat enum and export logic"
      - "Tamper-evident export with chain hashes"
    acceptance_criteria:
      - "Query by time range returns ordered results"
      - "Query by session_id filters correctly"
      - "limit and offset for pagination work correctly"
      - "JSON export includes chain_hash field"
      - "CSV export has header row"
      - "Parquet export uses appropriate column types"
      - "Query of 10,000 events completes in <100ms"
```

---

## Resource Layer: Quota Management

```yaml
  # ============================================================
  # RESOURCE: Quota Management
  # ============================================================

  - id: "TASK-13-020"
    title: "Implement ResourceQuotaManager with RAII Guards"
    description: |
      Create resource quota management for memory, CPU, and storage limits.

      ResourceType enum:
      - Memory, CpuTime, Storage, ConcurrentOperations, NetworkBandwidth, DatabaseConnections

      QuotaConfig struct:
      - resource: ResourceType
      - limit: u64
      - unit: ResourceUnit (Bytes, Milliseconds, Count, BytesPerSecond)
      - window: Option<Duration> (for time-windowed quotas)
      - soft_limit_percent: u8 (warn threshold, default 80%)

      Default quotas from spec:
      | Resource | Session | User | Global |
      |----------|---------|------|--------|
      | Memory | 256 MB | 1 GB | 16 GB |
      | CPU Time | 30 sec/req | 300 sec/min | 3600 sec/min |
      | Storage | 100 MB | 1 GB | 100 GB |
      | Concurrent | 10 | 50 | 500 |

      ResourceGuard: RAII wrapper that releases allocation on drop.
    layer: "resource"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-mcp-hardening/src/quota/manager.rs"
    requirements_traced:
      - REQ-MCPH-007
    dependencies: []
    input_dependencies: []
    output_deliverables:
      - "ResourceType and ResourceUnit enums"
      - "QuotaConfig struct with defaults"
      - "ResourceQuotaManager trait"
      - "DefaultQuotaManager implementation"
      - "ResourceGuard RAII type"
    acceptance_criteria:
      - "check_quota returns QuotaCheckResult in <100us"
      - "allocate returns ResourceAllocation handle"
      - "ResourceGuard releases on drop (even on panic)"
      - "Soft limit warning at 80% utilization"
      - "Hard limit blocks at 100% utilization"
      - "Default quotas match spec values"
      - "Integration test: allocate to limit then verify blocked"

  - id: "TASK-13-021"
    title: "Implement ResourceUsage Tracking and Trends"
    description: |
      Create resource usage tracking with trend analysis for proactive alerts.

      ResourceUsage struct:
      - resource: ResourceType
      - current: u64
      - limit: u64
      - unit: ResourceUnit
      - window_start: Option<Instant>
      - trend: UsageTrend

      UsageTrend enum:
      - Stable
      - Increasing { rate: f64 } (units per second)
      - Decreasing { rate: f64 }

      Trend calculation uses exponential moving average over 60 samples.
      Alert when trend predicts quota exhaustion within 5 minutes.
    layer: "resource"
    priority: "medium"
    estimated_hours: 3
    file_path: "crates/context-graph-mcp-hardening/src/quota/usage.rs"
    requirements_traced:
      - REQ-MCPH-007
    dependencies:
      - TASK-13-020
    input_dependencies:
      - "ResourceType from TASK-13-020"
    output_deliverables:
      - "ResourceUsage struct"
      - "UsageTrend enum"
      - "Trend calculation with EMA"
      - "Exhaustion prediction"
    acceptance_criteria:
      - "Trend correctly identifies increasing/decreasing usage"
      - "EMA uses 60 sample window"
      - "Exhaustion prediction triggers at 5 minute horizon"
      - "Stable trend when variance < 5%"
      - "Unit test with synthetic usage patterns"
```

---

## Integration Layer: SecurityMiddleware and MCP Tools

```yaml
  # ============================================================
  # INTEGRATION: Tower Middleware and MCP Tools
  # ============================================================

  - id: "TASK-13-022"
    title: "Implement SecurityMiddleware Tower Layer"
    description: |
      Create Tower-pattern security middleware composing all security components.

      SecurityLayer struct holds Arc references to:
      - InputValidator
      - RateLimiter
      - AuthManager
      - AuditLogger
      - ResourceQuotaManager

      impl Layer<S> for SecurityLayer:
      - type Service = SecurityMiddleware<S>
      - layer(&self, inner: S) -> SecurityMiddleware<S>

      SecurityMiddleware impl Service:
      1. Input validation (reject if invalid)
      2. Rate limiting (reject if exceeded)
      3. Authentication (reject if unauthorized)
      4. Resource quota check (reject if exceeded)
      5. Allocate resources
      6. Execute inner service
      7. Release resources
      8. Audit log success/failure

      All rejections logged to audit with appropriate event types.
    layer: "integration"
    priority: "critical"
    estimated_hours: 6
    file_path: "crates/context-graph-mcp-hardening/src/middleware/security.rs"
    requirements_traced:
      - REQ-MCPH-001
      - REQ-MCPH-002
      - REQ-MCPH-003
      - REQ-MCPH-004
      - REQ-MCPH-005
      - REQ-MCPH-006
      - REQ-MCPH-007
    dependencies:
      - TASK-13-005
      - TASK-13-011
      - TASK-13-015
      - TASK-13-018
      - TASK-13-020
    input_dependencies:
      - "InputValidator from TASK-13-005"
      - "RateLimiter from TASK-13-011"
      - "AuthManager from TASK-13-015"
      - "AuditLogger from TASK-13-018"
      - "ResourceQuotaManager from TASK-13-020"
    output_deliverables:
      - "SecurityLayer struct"
      - "SecurityMiddleware<S> struct"
      - "SecureMcpRequest and SecureMcpResponse types"
      - "SecurityContext for passing through pipeline"
    acceptance_criteria:
      - "Full pipeline executes in <5ms for valid requests"
      - "Each rejection produces audit event"
      - "Resources released on error paths"
      - "SecurityContext propagates through all layers"
      - "Integration test: full pipeline with valid/invalid requests"
      - "Fail-secure: component failure blocks request"

  - id: "TASK-13-023"
    title: "Implement Marblestone MCP Tool Handlers"
    description: |
      Create secured MCP tool handlers for the three Marblestone tools.

      Tool implementations:

      1. get_steering_feedback(session_id: &str) -> Result<SteeringFeedback>
         - Permission: Elevated (Level 2)
         - Validate session_id is UUID
         - Call steering service
         - Return SteeringFeedback with direction_vector, confidence, constraints

      2. omni_infer(direction: &str, clamped: Vec<ClampedValue>) -> Result<InferenceResult>
         - Permission: Elevated (Level 2)
         - Rate limit: 50/min per session
         - Validate direction length 1-256
         - Validate all ClampedValue.value in [min, max]
         - Call inference engine
         - Return InferenceResult with outputs, confidence, token_usage

      3. verify_code_node(node_id: Uuid) -> Result<VerificationStatus>
         - Permission: Admin (Level 3)
         - Validate node exists
         - Run verification pipeline
         - Sign result with verifier key
         - Return VerificationStatus with checks_passed, checks_failed
    layer: "integration"
    priority: "critical"
    estimated_hours: 5
    file_path: "crates/context-graph-mcp-hardening/src/tools/marblestone.rs"
    requirements_traced:
      - REQ-MCPH-008
      - REQ-MCPH-009
      - REQ-MCPH-010
    dependencies:
      - TASK-13-007
      - TASK-13-014
      - TASK-13-022
    input_dependencies:
      - "Marblestone schemas from TASK-13-007"
      - "Tool permissions from TASK-13-014"
      - "SecurityMiddleware from TASK-13-022"
    output_deliverables:
      - "get_steering_feedback handler"
      - "omni_infer handler with clamping"
      - "verify_code_node handler with signing"
      - "SteeringFeedback, InferenceResult, VerificationStatus types"
    acceptance_criteria:
      - "get_steering_feedback rejects non-Elevated sessions"
      - "omni_infer enforces 50/min rate limit"
      - "omni_infer clamps values to [min, max]"
      - "verify_code_node rejects non-Admin sessions"
      - "verify_code_node signatures are verifiable"
      - "All handlers log to audit trail"
      - "Integration test: each tool with valid/invalid permissions"

  - id: "TASK-13-024"
    title: "Implement SecurityError Type and Pipeline Integration"
    description: |
      Create comprehensive security error type and wire up the complete
      security pipeline for MCP server integration.

      SecurityError enum:
      - ValidationFailed(Vec<ValidationError>)
      - RateLimited { retry_after: Duration }
      - Unauthorized { required: PermissionLevel, actual: PermissionLevel }
      - QuotaExceeded { resource: ResourceType, usage: ResourceUsage }
      - ServiceUnavailable
      - ExecutionFailed
      - AuditError(AuditError)
      - QuotaError(QuotaError)

      Pipeline integration:
      - Create SecurityPipeline facade that wraps SecurityMiddleware
      - Provide builder pattern for configuration
      - Export metrics to Prometheus
      - Health check endpoint for liveness/readiness

      Wire into existing MCP server from Module 2.
    layer: "integration"
    priority: "critical"
    estimated_hours: 4
    file_path: "crates/context-graph-mcp-hardening/src/pipeline.rs"
    requirements_traced:
      - REQ-MCPH-001
      - REQ-MCPH-002
      - REQ-MCPH-003
      - REQ-MCPH-004
      - REQ-MCPH-005
      - REQ-MCPH-006
      - REQ-MCPH-007
    dependencies:
      - TASK-13-022
      - TASK-13-023
    input_dependencies:
      - "SecurityMiddleware from TASK-13-022"
      - "All component traits"
    output_deliverables:
      - "SecurityError enum"
      - "SecurityPipelineBuilder"
      - "Prometheus metrics export"
      - "Health check endpoint"
      - "MCP server integration"
    acceptance_criteria:
      - "All error variants map to appropriate HTTP status codes"
      - "Builder allows configuring all components"
      - "/metrics returns Prometheus format"
      - "/health returns OK when all components healthy"
      - "MCP server uses SecurityMiddleware for all requests"
      - "End-to-end test: MCP request through full security pipeline"
```

---

## Task Summary

| Layer | Tasks | Priority Distribution | Estimated Hours |
|-------|-------|----------------------|-----------------|
| Foundation | TASK-13-001 to TASK-13-008 | 4 Critical, 3 High, 1 Medium | 24 |
| Rate Limiting | TASK-13-009 to TASK-13-012 | 2 Critical, 1 High, 1 Medium | 13 |
| Authentication | TASK-13-013 to TASK-13-016 | 2 Critical, 2 High | 12 |
| Audit | TASK-13-017 to TASK-13-019 | 2 Critical, 1 High | 12 |
| Resource | TASK-13-020 to TASK-13-021 | 1 Critical, 1 Medium | 8 |
| Integration | TASK-13-022 to TASK-13-024 | 3 Critical | 15 |
| **Total** | **24 Tasks** | **14 Critical, 7 High, 3 Medium** | **84 Hours** |

---

## Dependency Graph

```
TASK-13-001 (ParameterSchema)
    |
    v
TASK-13-002 (ValidationResult) --> TASK-13-003 (Patterns) --> TASK-13-005 (InputValidator)
    |                                     |                           |
    v                                     v                           v
TASK-13-004 (Sanitizers) ----------------+                   TASK-13-007 (Marblestone Schemas)
    |                                                                 |
    v                                                                 |
TASK-13-008 (ValidatorError)                                         |
                                                                      |
TASK-13-009 (TokenBucket)                                             |
    |                                                                 |
    v                                                                 |
TASK-13-010 (RateLimitScope) --> TASK-13-011 (RateLimiter)           |
    |                                   |                             |
    v                                   v                             |
TASK-13-012 (Rate Metrics)              |                             |
                                        |                             |
TASK-13-013 (SessionToken)              |                             |
    |                                   |                             |
    v                                   v                             |
TASK-13-014 (ToolPermission) --> TASK-13-015 (AuthManager)           |
    |                                   |                             |
    v                                   v                             |
TASK-13-016 (AuthError)                 |                             |
                                        |                             |
TASK-13-017 (AuditEvent)                |                             |
    |                                   |                             |
    v                                   v                             |
TASK-13-018 (AuditLogger) --> TASK-13-019 (AuditQuery)               |
    |                                                                 |
    v                                                                 |
TASK-13-020 (QuotaManager) --> TASK-13-021 (ResourceUsage)           |
    |                                                                 |
    +----------------------------------------------------------------+
    |
    v
TASK-13-022 (SecurityMiddleware) --> TASK-13-023 (MCP Tools) --> TASK-13-024 (Pipeline)
```

---

## Requirements Traceability Matrix

| Requirement | Description | Tasks |
|-------------|-------------|-------|
| REQ-MCPH-001 | Input validation <1ms | TASK-13-001, 002, 004, 005, 006, 008 |
| REQ-MCPH-002 | Injection detection >99% | TASK-13-002, 003, 004, 005 |
| REQ-MCPH-003 | Rate limiting 1000/min global, 100/min session | TASK-13-009, 010, 011, 012 |
| REQ-MCPH-004 | Session authentication <500us | TASK-13-013, 015, 016 |
| REQ-MCPH-005 | Permission levels 0-4 | TASK-13-013, 014, 015 |
| REQ-MCPH-006 | Audit logging <200us | TASK-13-017, 018, 019 |
| REQ-MCPH-007 | Resource quotas | TASK-13-020, 021 |
| REQ-MCPH-008 | get_steering_feedback tool | TASK-13-007, 014, 023 |
| REQ-MCPH-009 | omni_infer tool | TASK-13-007, 014, 023 |
| REQ-MCPH-010 | verify_code_node tool | TASK-13-007, 014, 023 |

---

## Testing Requirements

1. **Unit Tests**: Each task includes component-level unit tests
2. **Integration Tests**: Pipeline tests for complete request flow
3. **Fuzz Tests**: Input validation with fuzzing (cargo-fuzz)
4. **Load Tests**: Rate limiting accuracy under load
5. **Security Tests**: Penetration testing for injection prevention
6. **Performance Tests**: Latency benchmarks for all components

---

## Notes

- All tasks are atomic with single responsibility
- No mock data implementations - all production-ready
- No fallbacks or workarounds - proper error handling only
- Performance targets are P99 latencies
- Security-first design with defense-in-depth
