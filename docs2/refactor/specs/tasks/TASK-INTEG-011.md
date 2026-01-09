# TASK-INTEG-011: Security Implementation (SEC-01 through SEC-08)

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-INTEG-011 |
| **Title** | Security Implementation (SEC-01 through SEC-08) |
| **Status** | :white_circle: todo |
| **Layer** | Integration |
| **Sequence** | 31 |
| **Estimated Days** | 4 |
| **Complexity** | High |

## Implements

- **SEC-01**: Input validation
- **SEC-02**: PII detection
- **SEC-03**: Rate limiting
- **SEC-04**: Session authentication
- **SEC-05**: Authorization
- **SEC-06**: Secrets management
- **SEC-07**: Subagent isolation
- **SEC-08**: Security logging

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-001 | MCP handlers to secure |
| TASK-INTEG-004 | Hook system for security events |

## Objective

Implement all security requirements from the constitution including input validation, PII detection, rate limiting, authentication, authorization, secrets management, subagent isolation, and security logging.

## Context

**Constitution Security Requirements (SEC-01 through SEC-08, lines 461-516):**

The constitution mandates comprehensive security controls that are currently **NOT COVERED** by any existing INTEG task. This is a critical gap.

## Scope

### In Scope

- **SEC-01**: Input validation and sanitization for all MCP tool inputs
- **SEC-02**: PII detection patterns (SSN, credit cards, emails, phones, addresses)
- **SEC-03**: Rate limiting per tool per session
- **SEC-04**: Session authentication with token expiry
- **SEC-05**: Authorization enforcement per tool
- **SEC-06**: Environment variable secrets management
- **SEC-07**: Subagent memory isolation
- **SEC-08**: Security event logging

### Out of Scope

- Network-level security (firewall, TLS termination)
- Audit compliance reporting
- Multi-tenant isolation

## Definition of Done

### SEC-01: Input Validation

```rust
// crates/context-graph-mcp/src/security/input_validation.rs

use regex::Regex;

/// Input validator for MCP tool parameters
pub struct InputValidator {
    max_text_length: usize,
    max_array_length: usize,
    forbidden_patterns: Vec<Regex>,
}

impl InputValidator {
    pub fn new(config: ValidationConfig) -> Self;

    /// Validate and sanitize text input
    pub fn validate_text(&self, input: &str) -> ValidationResult<String>;

    /// Validate numeric input within bounds
    pub fn validate_number<T: Num>(&self, input: T, min: T, max: T) -> ValidationResult<T>;

    /// Validate UUID format
    pub fn validate_uuid(&self, input: &str) -> ValidationResult<Uuid>;

    /// Validate JSON structure
    pub fn validate_json(&self, input: &serde_json::Value) -> ValidationResult<()>;

    /// Sanitize string for storage (escape, normalize)
    pub fn sanitize(&self, input: &str) -> String;
}

#[derive(Debug)]
pub struct ValidationConfig {
    pub max_text_length: usize,      // Default: 100_000
    pub max_array_length: usize,     // Default: 1_000
    pub reject_html: bool,           // Default: true
    pub reject_null_bytes: bool,     // Default: true
}
```

### SEC-02: PII Detection

```rust
// crates/context-graph-mcp/src/security/pii_detection.rs

/// PII detection with configurable patterns
pub struct PiiDetector {
    patterns: Vec<(PiiType, Regex)>,
    action: PiiAction,
}

#[derive(Debug, Clone, Copy)]
pub enum PiiType {
    Ssn,
    CreditCard,
    Email,
    PhoneNumber,
    Address,
    DateOfBirth,
    DriversLicense,
}

#[derive(Debug, Clone, Copy)]
pub enum PiiAction {
    Reject,     // Reject input entirely
    Mask,       // Replace PII with [REDACTED]
    Log,        // Log detection but allow
}

impl PiiDetector {
    pub fn new(action: PiiAction) -> Self;

    /// Detect PII in text, return types found
    pub fn detect(&self, text: &str) -> Vec<PiiMatch>;

    /// Apply action (mask or reject)
    pub fn process(&self, text: &str) -> PiiResult<String>;

    /// Check if text contains any PII
    pub fn contains_pii(&self, text: &str) -> bool;
}

#[derive(Debug)]
pub struct PiiMatch {
    pub pii_type: PiiType,
    pub start: usize,
    pub end: usize,
    pub matched: String,
}
```

### SEC-03: Rate Limiting

```rust
// crates/context-graph-mcp/src/security/rate_limiter.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Per-session, per-tool rate limiter
pub struct RateLimiter {
    limits: HashMap<String, RateLimit>,  // tool -> limit
    buckets: RwLock<HashMap<(SessionId, String), TokenBucket>>,
}

/// Rate limit configuration per tool (from constitution)
pub fn default_rate_limits() -> HashMap<String, RateLimit> {
    let mut limits = HashMap::new();
    limits.insert("inject_context".into(), RateLimit::new(100, Duration::from_secs(60)));
    limits.insert("store_memory".into(), RateLimit::new(50, Duration::from_secs(60)));
    limits.insert("search_graph".into(), RateLimit::new(200, Duration::from_secs(60)));
    limits.insert("discover_goals".into(), RateLimit::new(10, Duration::from_secs(60)));
    limits.insert("consolidate_memories".into(), RateLimit::new(1, Duration::from_secs(60)));
    limits
}

impl RateLimiter {
    pub fn new(limits: HashMap<String, RateLimit>) -> Self;

    /// Check if request is allowed (token bucket algorithm)
    pub fn check(&self, session: SessionId, tool: &str) -> RateLimitResult;

    /// Consume a token (call after allowing request)
    pub fn consume(&self, session: SessionId, tool: &str);

    /// Get remaining tokens for session/tool
    pub fn remaining(&self, session: SessionId, tool: &str) -> usize;

    /// Reset rate limit for session (admin)
    pub fn reset(&self, session: SessionId);
}

#[derive(Debug)]
pub struct RateLimit {
    pub max_requests: usize,
    pub window: Duration,
}
```

### SEC-04: Session Authentication

```rust
// crates/context-graph-mcp/src/security/auth.rs

use chrono::{DateTime, Utc, Duration as ChronoDuration};

/// Session authentication manager
pub struct SessionAuth {
    token_expiry: ChronoDuration,  // Default: 24 hours
    sessions: RwLock<HashMap<SessionToken, Session>>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub token: SessionToken,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub permissions: Permissions,
}

impl SessionAuth {
    pub fn new(token_expiry_hours: i64) -> Self;

    /// Create new session with token
    pub fn create_session(&self, permissions: Permissions) -> Session;

    /// Validate token, return session if valid
    pub fn validate(&self, token: &SessionToken) -> AuthResult<Session>;

    /// Invalidate session (logout)
    pub fn invalidate(&self, token: &SessionToken);

    /// Refresh session expiry
    pub fn refresh(&self, token: &SessionToken) -> AuthResult<Session>;

    /// Clean up expired sessions
    pub fn cleanup_expired(&self);
}
```

### SEC-05: Authorization

```rust
// crates/context-graph-mcp/src/security/authz.rs

/// Tool-level authorization
pub struct Authorizer {
    tool_permissions: HashMap<String, Permission>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Read,           // inject_context, search_graph
    Write,          // store_memory
    Admin,          // consolidate_memories
    GoalDiscovery,  // discover_goals (read-only by default)
}

impl Authorizer {
    pub fn new() -> Self;

    /// Check if session has permission for tool
    pub fn authorize(&self, session: &Session, tool: &str) -> AuthzResult<()>;

    /// Get required permission for tool
    pub fn required_permission(&self, tool: &str) -> Permission;
}
```

### SEC-06: Secrets Management

```rust
// crates/context-graph-mcp/src/security/secrets.rs

/// Environment-based secrets management
pub struct SecretsManager;

impl SecretsManager {
    /// Get database path from env
    pub fn db_path() -> Result<PathBuf, ConfigError> {
        std::env::var("CONTEXT_GRAPH_DB_PATH")
            .map(PathBuf::from)
            .map_err(|_| ConfigError::EnvNotSet("CONTEXT_GRAPH_DB_PATH".into()))
    }

    /// Get model directory from env
    pub fn model_dir() -> Result<PathBuf, ConfigError> {
        std::env::var("CONTEXT_GRAPH_MODEL_DIR")
            .map(PathBuf::from)
            .map_err(|_| ConfigError::EnvNotSet("CONTEXT_GRAPH_MODEL_DIR".into()))
    }

    /// Required env vars
    pub fn required_vars() -> Vec<&'static str> {
        vec![
            "CONTEXT_GRAPH_DB_PATH",
            "CONTEXT_GRAPH_MODEL_DIR",
        ]
    }

    /// Validate all required vars are set
    pub fn validate_environment() -> Result<(), ConfigError>;
}
```

### SEC-07: Subagent Isolation

```rust
// crates/context-graph-mcp/src/security/isolation.rs

/// Subagent memory isolation
pub struct IsolationManager {
    session_scopes: RwLock<HashMap<SessionId, MemoryScope>>,
}

#[derive(Debug, Clone)]
pub struct MemoryScope {
    pub session_id: SessionId,
    pub allowed_namespaces: Vec<String>,
    pub cross_session_access: bool,
}

impl IsolationManager {
    /// Create isolated scope for subagent
    pub fn create_scope(&self, parent_session: SessionId) -> MemoryScope;

    /// Check if access to memory is allowed
    pub fn check_access(&self, scope: &MemoryScope, memory_id: Uuid) -> bool;

    /// Grant cross-session access (explicit consolidation)
    pub fn grant_cross_session(&self, scope: &mut MemoryScope);
}
```

### SEC-08: Security Logging

```rust
// crates/context-graph-mcp/src/security/audit_log.rs

use serde::Serialize;
use tracing::{info, warn, error};

/// Security event types
#[derive(Debug, Clone, Serialize)]
pub enum SecurityEvent {
    AuthenticationFailure { session_token: String, reason: String },
    RateLimitExceeded { session_id: String, tool: String, limit: usize },
    PiiDetected { session_id: String, pii_type: String, action_taken: String },
    InvalidInputRejected { session_id: String, tool: String, reason: String },
    AuthorizationDenied { session_id: String, tool: String, required: String },
    SessionExpired { session_id: String },
}

/// Structured security audit logger
pub struct SecurityAuditLog;

impl SecurityAuditLog {
    /// Log security event with structured JSON
    pub fn log(event: SecurityEvent) {
        let json = serde_json::to_string(&event).unwrap();
        match &event {
            SecurityEvent::AuthenticationFailure { .. } => warn!(target: "security", "{}", json),
            SecurityEvent::RateLimitExceeded { .. } => warn!(target: "security", "{}", json),
            SecurityEvent::PiiDetected { .. } => info!(target: "security", "{}", json),
            SecurityEvent::InvalidInputRejected { .. } => info!(target: "security", "{}", json),
            SecurityEvent::AuthorizationDenied { .. } => warn!(target: "security", "{}", json),
            SecurityEvent::SessionExpired { .. } => info!(target: "security", "{}", json),
        }
    }
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Input validation latency | < 1ms |
| PII detection latency | < 5ms |
| Rate limit check | < 0.1ms |
| Auth validation | < 1ms |

## Verification

- [ ] All MCP tool inputs validated before processing
- [ ] PII patterns detect SSN, credit cards, emails, phones
- [ ] Rate limits enforced per constitution (100/50/200/1 per min)
- [ ] Session tokens expire after 24 hours
- [ ] Unauthorized tool access returns 403
- [ ] Security events logged in JSON format
- [ ] Subagent memory isolated from other sessions

## Files to Create

| File | Purpose |
|------|---------|
| `crates/context-graph-mcp/src/security/mod.rs` | Security module root |
| `crates/context-graph-mcp/src/security/input_validation.rs` | SEC-01 |
| `crates/context-graph-mcp/src/security/pii_detection.rs` | SEC-02 |
| `crates/context-graph-mcp/src/security/rate_limiter.rs` | SEC-03 |
| `crates/context-graph-mcp/src/security/auth.rs` | SEC-04 |
| `crates/context-graph-mcp/src/security/authz.rs` | SEC-05 |
| `crates/context-graph-mcp/src/security/secrets.rs` | SEC-06 |
| `crates/context-graph-mcp/src/security/isolation.rs` | SEC-07 |
| `crates/context-graph-mcp/src/security/audit_log.rs` | SEC-08 |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PII bypass | Medium | Critical | Multiple pattern sources |
| Rate limit bypass | Low | Medium | Token bucket algorithm |
| Token leak | Low | Critical | Short expiry, secure generation |

## Traceability

- Source: Constitution security_requirements (lines 461-516)
