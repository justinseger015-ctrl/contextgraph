# Module 13: MCP Hardening Technical Specification

**Version**: 1.0.0
**Status**: Draft
**Last Updated**: 2025-12-31
**Requirements**: REQ-MCPH-001 through REQ-MCPH-010

---

## 1. Overview

### 1.1 Purpose

This module provides comprehensive security hardening for the Model Context Protocol (MCP) layer, implementing defense-in-depth strategies including input validation, rate limiting, authentication, audit logging, resource quotas, and security middleware.

### 1.2 Scope

- Input validation and schema enforcement
- Rate limiting with token bucket algorithm
- Session-based authentication and authorization
- Tamper-evident async audit logging
- Resource quota management
- Tower-pattern security middleware

### 1.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MCP Request Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Request    │→│   Security   │→│    Input     │              │
│  │   Ingress    │  │  Middleware  │  │  Validator   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                │                  │                       │
│         ▼                ▼                  ▼                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ RateLimiter  │→│  AuthManager │→│   Resource   │              │
│  │ (TokenBucket)│  │  (Sessions)  │  │ QuotaManager │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                │                  │                       │
│         └────────────────┼──────────────────┘                       │
│                          ▼                                          │
│                 ┌──────────────┐                                    │
│                 │ AuditLogger  │──→ Tamper-Evident Log             │
│                 │  (<200μs)    │                                    │
│                 └──────────────┘                                    │
│                          │                                          │
│                          ▼                                          │
│                 ┌──────────────┐                                    │
│                 │  MCP Tool    │                                    │
│                 │  Execution   │                                    │
│                 └──────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 InputValidator (REQ-MCPH-001/002)

#### 2.1.1 Purpose

Validates all incoming MCP requests against predefined schemas and prevents injection attacks.

#### 2.1.2 Interface Definition

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Schema definition for MCP tool parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub name: String,
    pub param_type: ParamType,
    pub required: bool,
    pub constraints: Vec<Constraint>,
    pub sanitizers: Vec<SanitizerType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamType {
    String { min_len: usize, max_len: usize },
    Integer { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Boolean,
    Uuid,
    Array { item_type: Box<ParamType>, max_items: usize },
    Object { schema: HashMap<String, ParameterSchema> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    Regex(String),
    AllowedValues(Vec<String>),
    NoSqlInjection,
    NoXss,
    NoCommandInjection,
    NoPathTraversal,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SanitizerType {
    HtmlEscape,
    SqlEscape,
    ShellEscape,
    PathNormalize,
    UnicodeNormalize,
    Trim,
}

/// Validation result with detailed error information
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub sanitized_input: Option<serde_json::Value>,
    pub threat_indicators: Vec<ThreatIndicator>,
}

#[derive(Debug)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub code: ValidationErrorCode,
}

#[derive(Debug)]
pub enum ValidationErrorCode {
    MissingRequired,
    TypeMismatch,
    ConstraintViolation,
    InjectionDetected,
    SchemaMissing,
}

#[derive(Debug)]
pub struct ThreatIndicator {
    pub threat_type: ThreatType,
    pub confidence: f32,
    pub payload_snippet: String,
}

#[derive(Debug)]
pub enum ThreatType {
    SqlInjection,
    XssAttempt,
    CommandInjection,
    PathTraversal,
    BufferOverflow,
    FormatString,
}

/// Main validator trait
#[async_trait::async_trait]
pub trait InputValidator: Send + Sync {
    /// Register a schema for an MCP tool
    async fn register_schema(&self, tool_name: &str, schema: Vec<ParameterSchema>) -> Result<(), ValidatorError>;

    /// Validate input against registered schema
    async fn validate(&self, tool_name: &str, input: &serde_json::Value) -> ValidationResult;

    /// Check for injection patterns without full validation
    async fn detect_injection(&self, input: &str) -> Vec<ThreatIndicator>;

    /// Sanitize input according to schema
    async fn sanitize(&self, tool_name: &str, input: serde_json::Value) -> Result<serde_json::Value, ValidatorError>;
}
```

#### 2.1.3 Injection Prevention Patterns

```rust
/// SQL injection detection patterns
const SQL_INJECTION_PATTERNS: &[&str] = &[
    r"(?i)(\b(union|select|insert|update|delete|drop|truncate|exec|execute)\b)",
    r"(?i)(--|\#|\/\*)",
    r"(?i)(\bor\b|\band\b)\s*[\d\w]+\s*=\s*[\d\w]+",
    r"(?i)('\s*(or|and)\s*')",
    r"(?i)(;\s*(drop|delete|truncate|update|insert))",
    r"(?i)(0x[0-9a-f]+)",
    r"(?i)(char\s*\(|concat\s*\()",
];

/// XSS detection patterns
const XSS_PATTERNS: &[&str] = &[
    r"(?i)<script[^>]*>",
    r"(?i)javascript\s*:",
    r"(?i)on(load|error|click|mouse|focus|blur|change|submit)\s*=",
    r"(?i)<iframe[^>]*>",
    r"(?i)expression\s*\(",
    r"(?i)data\s*:\s*text/html",
];

/// Command injection patterns
const COMMAND_INJECTION_PATTERNS: &[&str] = &[
    r"[;&|`$]",
    r"\$\(.*\)",
    r"`.*`",
    r"(?i)(eval|exec|system|passthru|shell_exec)",
    r"(?i)(\|\||&&)",
    r"(?i)(>|>>|<)\s*[/\w]",
];

/// Path traversal patterns
const PATH_TRAVERSAL_PATTERNS: &[&str] = &[
    r"\.\.[/\\]",
    r"(?i)%2e%2e[/\\%]",
    r"(?i)%252e%252e",
    r"(?i)\.\.%c0%af",
    r"(?i)\.\.%c1%9c",
];
```

---

### 2.2 RateLimiter (REQ-MCPH-003)

#### 2.2.1 Purpose

Implements token bucket rate limiting with configurable global and per-session limits.

#### 2.2.2 Configuration

| Scope | Limit | Burst | Refill Rate |
|-------|-------|-------|-------------|
| Global | 1000 req/min | 100 | 16.67/sec |
| Session | 100 req/min | 20 | 1.67/sec |
| Tool-specific | Configurable | Configurable | Configurable |

#### 2.2.3 Interface Definition

```rust
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Token bucket configuration
#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub capacity: u64,
    pub refill_rate: f64,  // tokens per second
    pub burst_size: u64,
}

/// Rate limit scope
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum RateLimitScope {
    Global,
    Session(String),
    User(Uuid),
    Tool(String),
    IpAddress(std::net::IpAddr),
    Composite(Vec<RateLimitScope>),
}

/// Rate limit decision
#[derive(Debug)]
pub struct RateLimitDecision {
    pub allowed: bool,
    pub tokens_remaining: u64,
    pub retry_after: Option<Duration>,
    pub scope: RateLimitScope,
    pub quota_info: QuotaInfo,
}

#[derive(Debug)]
pub struct QuotaInfo {
    pub limit: u64,
    pub remaining: u64,
    pub reset_at: Instant,
    pub window_seconds: u64,
}

/// Token bucket state
#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    config: BucketConfig,
}

impl TokenBucket {
    fn new(config: BucketConfig) -> Self {
        Self {
            tokens: config.capacity as f64,
            last_refill: Instant::now(),
            config,
        }
    }

    fn try_consume(&mut self, tokens: u64) -> bool {
        self.refill();
        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.config.refill_rate)
            .min(self.config.capacity as f64);
        self.last_refill = now;
    }

    fn time_until_available(&self, tokens: u64) -> Duration {
        let needed = (tokens as f64 - self.tokens).max(0.0);
        Duration::from_secs_f64(needed / self.config.refill_rate)
    }
}

/// Rate limiter trait
#[async_trait::async_trait]
pub trait RateLimiter: Send + Sync {
    /// Check if request is allowed without consuming tokens
    async fn check(&self, scope: &RateLimitScope) -> RateLimitDecision;

    /// Consume tokens and return decision
    async fn acquire(&self, scope: &RateLimitScope, tokens: u64) -> RateLimitDecision;

    /// Configure bucket for scope
    async fn configure(&self, scope: RateLimitScope, config: BucketConfig);

    /// Reset bucket for scope
    async fn reset(&self, scope: &RateLimitScope);

    /// Get current quota information
    async fn quota(&self, scope: &RateLimitScope) -> QuotaInfo;
}

/// Default rate limit configurations
pub fn default_global_config() -> BucketConfig {
    BucketConfig {
        capacity: 1000,
        refill_rate: 1000.0 / 60.0,  // 1000 per minute
        burst_size: 100,
    }
}

pub fn default_session_config() -> BucketConfig {
    BucketConfig {
        capacity: 100,
        refill_rate: 100.0 / 60.0,  // 100 per minute
        burst_size: 20,
    }
}
```

---

### 2.3 AuthManager (REQ-MCPH-004/005)

#### 2.3.1 Purpose

Manages session tokens and enforces permission levels for MCP tool access.

#### 2.3.2 Permission Model

```
Permission Levels:
├── Level 0: Public (read-only, no auth required)
├── Level 1: Authenticated (basic tool access)
├── Level 2: Elevated (steering tools, inference)
├── Level 3: Admin (verification, system tools)
└── Level 4: System (internal operations only)
```

#### 2.3.3 Interface Definition

```rust
use chrono::{DateTime, Utc};
use std::collections::HashSet;

/// Session token with embedded claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionToken {
    pub session_id: Uuid,
    pub user_id: Option<Uuid>,
    pub permission_level: PermissionLevel,
    pub granted_tools: HashSet<String>,
    pub denied_tools: HashSet<String>,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub refresh_token: Option<String>,
    pub metadata: SessionMetadata,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermissionLevel {
    Public = 0,
    Authenticated = 1,
    Elevated = 2,
    Admin = 3,
    System = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub client_ip: Option<std::net::IpAddr>,
    pub user_agent: Option<String>,
    pub device_fingerprint: Option<String>,
    pub mfa_verified: bool,
    pub last_activity: DateTime<Utc>,
}

/// Tool permission requirements
#[derive(Debug, Clone)]
pub struct ToolPermission {
    pub tool_name: String,
    pub required_level: PermissionLevel,
    pub additional_claims: Vec<String>,
    pub rate_limit_override: Option<BucketConfig>,
}

/// Authorization decision
#[derive(Debug)]
pub struct AuthDecision {
    pub allowed: bool,
    pub reason: Option<String>,
    pub session: Option<SessionToken>,
    pub required_level: PermissionLevel,
    pub actual_level: PermissionLevel,
}

/// Auth manager trait
#[async_trait::async_trait]
pub trait AuthManager: Send + Sync {
    /// Create new session
    async fn create_session(
        &self,
        user_id: Option<Uuid>,
        permission_level: PermissionLevel,
        ttl: Duration,
        metadata: SessionMetadata,
    ) -> Result<SessionToken, AuthError>;

    /// Validate session token
    async fn validate_session(&self, token: &str) -> Result<SessionToken, AuthError>;

    /// Check tool authorization
    async fn authorize_tool(
        &self,
        session: &SessionToken,
        tool_name: &str,
    ) -> AuthDecision;

    /// Refresh session token
    async fn refresh_session(&self, refresh_token: &str) -> Result<SessionToken, AuthError>;

    /// Revoke session
    async fn revoke_session(&self, session_id: Uuid) -> Result<(), AuthError>;

    /// Register tool permission requirements
    async fn register_tool_permission(&self, permission: ToolPermission);

    /// Elevate session permissions (requires MFA)
    async fn elevate_session(
        &self,
        session_id: Uuid,
        new_level: PermissionLevel,
        mfa_token: &str,
    ) -> Result<SessionToken, AuthError>;
}

/// Tool permission registry
pub fn marblestone_tool_permissions() -> Vec<ToolPermission> {
    vec![
        ToolPermission {
            tool_name: "get_steering_feedback".into(),
            required_level: PermissionLevel::Elevated,
            additional_claims: vec!["steering:read".into()],
            rate_limit_override: None,
        },
        ToolPermission {
            tool_name: "omni_infer".into(),
            required_level: PermissionLevel::Elevated,
            additional_claims: vec!["inference:execute".into()],
            rate_limit_override: Some(BucketConfig {
                capacity: 50,
                refill_rate: 50.0 / 60.0,
                burst_size: 10,
            }),
        },
        ToolPermission {
            tool_name: "verify_code_node".into(),
            required_level: PermissionLevel::Admin,
            additional_claims: vec!["verification:execute".into()],
            rate_limit_override: None,
        },
    ]
}
```

---

### 2.4 AuditLogger (REQ-MCPH-006)

#### 2.4.1 Purpose

Provides tamper-evident, async audit logging with sub-200 microsecond latency.

#### 2.4.2 Performance Requirements

| Metric | Target | Maximum |
|--------|--------|---------|
| Log write latency | < 100us | < 200us |
| Buffer flush interval | 100ms | 500ms |
| Integrity check overhead | < 50us | < 100us |

#### 2.4.3 Interface Definition

```rust
use ring::digest::{Context, SHA256};
use std::sync::atomic::{AtomicU64, Ordering};

/// Audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: Uuid,
    pub sequence_number: u64,
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub session_id: Option<Uuid>,
    pub user_id: Option<Uuid>,
    pub tool_name: Option<String>,
    pub action: String,
    pub outcome: AuditOutcome,
    pub metadata: serde_json::Value,
    pub chain_hash: String,  // Hash of previous event for tamper evidence
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    ToolExecution,
    RateLimitHit,
    ValidationFailure,
    SecurityAlert,
    ConfigChange,
    SystemEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure { reason: String },
    Blocked { reason: String },
    Throttled,
}

/// Tamper-evident log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TamperEvidentEntry {
    pub event: AuditEvent,
    pub signature: String,
    pub merkle_proof: Option<Vec<String>>,
}

/// Audit query parameters
#[derive(Debug, Clone)]
pub struct AuditQuery {
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub event_types: Option<Vec<AuditEventType>>,
    pub session_id: Option<Uuid>,
    pub user_id: Option<Uuid>,
    pub tool_name: Option<String>,
    pub outcome: Option<AuditOutcome>,
    pub limit: usize,
    pub offset: usize,
}

/// Audit logger trait
#[async_trait::async_trait]
pub trait AuditLogger: Send + Sync {
    /// Log event asynchronously (must complete in <200us)
    async fn log(&self, event: AuditEvent) -> Result<u64, AuditError>;

    /// Log batch of events
    async fn log_batch(&self, events: Vec<AuditEvent>) -> Result<Vec<u64>, AuditError>;

    /// Query audit log
    async fn query(&self, query: AuditQuery) -> Result<Vec<TamperEvidentEntry>, AuditError>;

    /// Verify log integrity from sequence A to B
    async fn verify_integrity(&self, from_seq: u64, to_seq: u64) -> Result<bool, AuditError>;

    /// Export audit log segment for compliance
    async fn export(&self, query: AuditQuery, format: ExportFormat) -> Result<Vec<u8>, AuditError>;

    /// Get current sequence number
    fn current_sequence(&self) -> u64;
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Parquet,
}

/// Chain hash computation for tamper evidence
pub fn compute_chain_hash(prev_hash: &str, event: &AuditEvent) -> String {
    let mut context = Context::new(&SHA256);
    context.update(prev_hash.as_bytes());
    context.update(&event.sequence_number.to_le_bytes());
    context.update(event.event_id.as_bytes());
    context.update(event.timestamp.to_rfc3339().as_bytes());
    context.update(event.action.as_bytes());
    let digest = context.finish();
    hex::encode(digest.as_ref())
}
```

---

### 2.5 ResourceQuotaManager (REQ-MCPH-007)

#### 2.5.1 Purpose

Enforces resource limits for memory, CPU, and storage per session/user.

#### 2.5.2 Default Quotas

| Resource | Session Limit | User Limit | Global Limit |
|----------|--------------|------------|--------------|
| Memory | 256 MB | 1 GB | 16 GB |
| CPU Time | 30 sec/req | 300 sec/min | 3600 sec/min |
| Storage | 100 MB | 1 GB | 100 GB |
| Concurrent Ops | 10 | 50 | 500 |

#### 2.5.3 Interface Definition

```rust
/// Resource types
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum ResourceType {
    Memory,
    CpuTime,
    Storage,
    ConcurrentOperations,
    NetworkBandwidth,
    DatabaseConnections,
}

/// Quota configuration
#[derive(Debug, Clone)]
pub struct QuotaConfig {
    pub resource: ResourceType,
    pub limit: u64,
    pub unit: ResourceUnit,
    pub window: Option<Duration>,
    pub soft_limit_percent: u8,  // Warn at this percentage
}

#[derive(Debug, Clone, Copy)]
pub enum ResourceUnit {
    Bytes,
    Milliseconds,
    Count,
    BytesPerSecond,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub resource: ResourceType,
    pub current: u64,
    pub limit: u64,
    pub unit: ResourceUnit,
    pub window_start: Option<Instant>,
    pub trend: UsageTrend,
}

#[derive(Debug, Clone)]
pub enum UsageTrend {
    Stable,
    Increasing { rate: f64 },
    Decreasing { rate: f64 },
}

/// Quota check result
#[derive(Debug)]
pub struct QuotaCheckResult {
    pub allowed: bool,
    pub resource: ResourceType,
    pub usage: ResourceUsage,
    pub would_exceed_soft: bool,
    pub would_exceed_hard: bool,
    pub suggested_delay: Option<Duration>,
}

/// Resource allocation handle
#[derive(Debug)]
pub struct ResourceAllocation {
    pub allocation_id: Uuid,
    pub resource: ResourceType,
    pub amount: u64,
    pub scope: QuotaScope,
    pub created_at: Instant,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum QuotaScope {
    Session(String),
    User(Uuid),
    Tool(String),
    Global,
}

/// Resource quota manager trait
#[async_trait::async_trait]
pub trait ResourceQuotaManager: Send + Sync {
    /// Check if resource allocation is allowed
    async fn check_quota(
        &self,
        scope: &QuotaScope,
        resource: ResourceType,
        amount: u64,
    ) -> QuotaCheckResult;

    /// Allocate resource (returns handle for later release)
    async fn allocate(
        &self,
        scope: &QuotaScope,
        resource: ResourceType,
        amount: u64,
    ) -> Result<ResourceAllocation, QuotaError>;

    /// Release allocated resource
    async fn release(&self, allocation: ResourceAllocation);

    /// Get current usage for scope
    async fn get_usage(&self, scope: &QuotaScope) -> Vec<ResourceUsage>;

    /// Configure quota for scope
    async fn configure_quota(&self, scope: QuotaScope, config: QuotaConfig);

    /// Reset usage counters for scope
    async fn reset_usage(&self, scope: &QuotaScope, resource: Option<ResourceType>);
}

/// RAII guard for automatic resource release
pub struct ResourceGuard {
    allocation: Option<ResourceAllocation>,
    manager: Arc<dyn ResourceQuotaManager>,
}

impl Drop for ResourceGuard {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            let manager = self.manager.clone();
            tokio::spawn(async move {
                manager.release(allocation).await;
            });
        }
    }
}
```

---

### 2.6 SecurityMiddleware (REQ-MCPH-008/009/010)

#### 2.6.1 Purpose

Tower-pattern middleware composing all security components into a unified pipeline.

#### 2.6.2 Interface Definition

```rust
use tower::{Layer, Service};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

/// Security context passed through middleware chain
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub request_id: Uuid,
    pub session: Option<SessionToken>,
    pub client_ip: Option<std::net::IpAddr>,
    pub validated_input: Option<serde_json::Value>,
    pub rate_limit_info: Option<QuotaInfo>,
    pub resource_allocation: Option<Uuid>,
    pub audit_sequence: Option<u64>,
}

/// MCP request with security context
#[derive(Debug)]
pub struct SecureMcpRequest<T> {
    pub inner: T,
    pub security_context: SecurityContext,
    pub tool_name: String,
    pub parameters: serde_json::Value,
}

/// MCP response with audit info
#[derive(Debug)]
pub struct SecureMcpResponse<T> {
    pub inner: T,
    pub audit_sequence: u64,
    pub execution_time: Duration,
}

/// Security middleware layer
pub struct SecurityLayer {
    validator: Arc<dyn InputValidator>,
    rate_limiter: Arc<dyn RateLimiter>,
    auth_manager: Arc<dyn AuthManager>,
    audit_logger: Arc<dyn AuditLogger>,
    quota_manager: Arc<dyn ResourceQuotaManager>,
}

impl<S> Layer<S> for SecurityLayer {
    type Service = SecurityMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        SecurityMiddleware {
            inner,
            validator: self.validator.clone(),
            rate_limiter: self.rate_limiter.clone(),
            auth_manager: self.auth_manager.clone(),
            audit_logger: self.audit_logger.clone(),
            quota_manager: self.quota_manager.clone(),
        }
    }
}

/// Security middleware service
pub struct SecurityMiddleware<S> {
    inner: S,
    validator: Arc<dyn InputValidator>,
    rate_limiter: Arc<dyn RateLimiter>,
    auth_manager: Arc<dyn AuthManager>,
    audit_logger: Arc<dyn AuditLogger>,
    quota_manager: Arc<dyn ResourceQuotaManager>,
}

impl<S, Req, Res> Service<SecureMcpRequest<Req>> for SecurityMiddleware<S>
where
    S: Service<SecureMcpRequest<Req>, Response = Res> + Clone + Send + 'static,
    S::Future: Send,
    Req: Send + 'static,
    Res: Send + 'static,
{
    type Response = SecureMcpResponse<Res>;
    type Error = SecurityError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut TaskContext<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(|_| SecurityError::ServiceUnavailable)
    }

    fn call(&mut self, req: SecureMcpRequest<Req>) -> Self::Future {
        let inner = self.inner.clone();
        let validator = self.validator.clone();
        let rate_limiter = self.rate_limiter.clone();
        let auth_manager = self.auth_manager.clone();
        let audit_logger = self.audit_logger.clone();
        let quota_manager = self.quota_manager.clone();

        Box::pin(async move {
            let start = Instant::now();
            let mut ctx = req.security_context.clone();

            // 1. Input Validation
            let validation = validator.validate(&req.tool_name, &req.parameters).await;
            if !validation.valid {
                audit_logger.log(AuditEvent {
                    event_id: Uuid::new_v4(),
                    sequence_number: 0,
                    timestamp: Utc::now(),
                    event_type: AuditEventType::ValidationFailure,
                    session_id: ctx.session.as_ref().map(|s| s.session_id),
                    user_id: ctx.session.as_ref().and_then(|s| s.user_id),
                    tool_name: Some(req.tool_name.clone()),
                    action: "validate_input".into(),
                    outcome: AuditOutcome::Failure {
                        reason: format!("{:?}", validation.errors),
                    },
                    metadata: serde_json::json!({"threats": validation.threat_indicators}),
                    chain_hash: String::new(),
                }).await?;

                return Err(SecurityError::ValidationFailed(validation.errors));
            }
            ctx.validated_input = validation.sanitized_input;

            // 2. Rate Limiting
            let session_scope = ctx.session.as_ref()
                .map(|s| RateLimitScope::Session(s.session_id.to_string()))
                .unwrap_or(RateLimitScope::Global);

            let rate_decision = rate_limiter.acquire(&session_scope, 1).await;
            if !rate_decision.allowed {
                audit_logger.log(AuditEvent {
                    event_id: Uuid::new_v4(),
                    sequence_number: 0,
                    timestamp: Utc::now(),
                    event_type: AuditEventType::RateLimitHit,
                    session_id: ctx.session.as_ref().map(|s| s.session_id),
                    user_id: ctx.session.as_ref().and_then(|s| s.user_id),
                    tool_name: Some(req.tool_name.clone()),
                    action: "rate_limit_check".into(),
                    outcome: AuditOutcome::Throttled,
                    metadata: serde_json::json!(rate_decision.quota_info),
                    chain_hash: String::new(),
                }).await?;

                return Err(SecurityError::RateLimited {
                    retry_after: rate_decision.retry_after.unwrap_or(Duration::from_secs(60)),
                });
            }
            ctx.rate_limit_info = Some(rate_decision.quota_info);

            // 3. Authorization
            if let Some(ref session) = ctx.session {
                let auth_decision = auth_manager.authorize_tool(session, &req.tool_name).await;
                if !auth_decision.allowed {
                    audit_logger.log(AuditEvent {
                        event_id: Uuid::new_v4(),
                        sequence_number: 0,
                        timestamp: Utc::now(),
                        event_type: AuditEventType::Authorization,
                        session_id: Some(session.session_id),
                        user_id: session.user_id,
                        tool_name: Some(req.tool_name.clone()),
                        action: "authorize_tool".into(),
                        outcome: AuditOutcome::Blocked {
                            reason: auth_decision.reason.unwrap_or_default(),
                        },
                        metadata: serde_json::json!({
                            "required_level": auth_decision.required_level,
                            "actual_level": auth_decision.actual_level,
                        }),
                        chain_hash: String::new(),
                    }).await?;

                    return Err(SecurityError::Unauthorized {
                        required: auth_decision.required_level,
                        actual: auth_decision.actual_level,
                    });
                }
            }

            // 4. Resource Quota Check
            let quota_scope = ctx.session.as_ref()
                .map(|s| QuotaScope::Session(s.session_id.to_string()))
                .unwrap_or(QuotaScope::Global);

            let quota_check = quota_manager.check_quota(
                &quota_scope,
                ResourceType::ConcurrentOperations,
                1,
            ).await;

            if !quota_check.allowed {
                return Err(SecurityError::QuotaExceeded {
                    resource: quota_check.resource,
                    usage: quota_check.usage,
                });
            }

            let allocation = quota_manager.allocate(
                &quota_scope,
                ResourceType::ConcurrentOperations,
                1,
            ).await?;
            ctx.resource_allocation = Some(allocation.allocation_id);

            // 5. Execute Inner Service
            let mut inner_service = inner;
            let secured_req = SecureMcpRequest {
                inner: req.inner,
                security_context: ctx.clone(),
                tool_name: req.tool_name.clone(),
                parameters: ctx.validated_input.clone().unwrap_or(req.parameters),
            };

            let result = inner_service.call(secured_req).await;

            // 6. Release Resources
            quota_manager.release(allocation).await;

            // 7. Audit Success/Failure
            let execution_time = start.elapsed();
            let audit_seq = audit_logger.log(AuditEvent {
                event_id: Uuid::new_v4(),
                sequence_number: 0,
                timestamp: Utc::now(),
                event_type: AuditEventType::ToolExecution,
                session_id: ctx.session.as_ref().map(|s| s.session_id),
                user_id: ctx.session.as_ref().and_then(|s| s.user_id),
                tool_name: Some(req.tool_name),
                action: "execute_tool".into(),
                outcome: if result.is_ok() {
                    AuditOutcome::Success
                } else {
                    AuditOutcome::Failure { reason: "execution_error".into() }
                },
                metadata: serde_json::json!({
                    "execution_time_ms": execution_time.as_millis(),
                }),
                chain_hash: String::new(),
            }).await?;

            result.map(|res| SecureMcpResponse {
                inner: res,
                audit_sequence: audit_seq,
                execution_time,
            }).map_err(|_| SecurityError::ExecutionFailed)
        })
    }
}
```

---

## 3. Marblestone MCP Tools (REQ-MCPH-008/009/010)

### 3.1 get_steering_feedback

```rust
/// Steering feedback from the guidance system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringFeedback {
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    pub direction_vector: Vec<f32>,
    pub confidence: f32,
    pub constraints: Vec<SteeringConstraint>,
    pub recommendations: Vec<SteeringRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringConstraint {
    pub constraint_type: String,
    pub value: f64,
    pub binding: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringRecommendation {
    pub action: String,
    pub priority: u8,
    pub rationale: String,
}

/// MCP tool: get_steering_feedback
///
/// Retrieves steering feedback for a given session, providing guidance
/// on inference direction and constraints.
///
/// # Security
/// - Required permission: Elevated (Level 2)
/// - Rate limit: 100 req/min per session
/// - Audit: Full logging of all feedback requests
///
/// # Parameters
/// - session_id: Valid UUID session identifier
///
/// # Returns
/// SteeringFeedback containing direction vectors and recommendations
pub async fn get_steering_feedback(session_id: &str) -> Result<SteeringFeedback, McpError> {
    // Validate session_id format
    let session_uuid = Uuid::parse_str(session_id)
        .map_err(|_| McpError::InvalidParameter("session_id must be valid UUID"))?;

    // Implementation delegates to steering service
    let feedback = steering_service::get_feedback(session_uuid).await?;

    Ok(feedback)
}
```

### 3.2 omni_infer

```rust
/// Clamped value for safe inference boundaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClampedValue {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
}

impl ClampedValue {
    pub fn clamped(&self) -> f64 {
        self.value.clamp(self.min, self.max)
    }

    pub fn is_valid(&self) -> bool {
        self.min <= self.value && self.value <= self.max
    }
}

/// Inference result from omni-directional inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub inference_id: Uuid,
    pub direction: String,
    pub outputs: Vec<InferenceOutput>,
    pub confidence: f32,
    pub latency_ms: u64,
    pub token_usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub name: String,
    pub value: serde_json::Value,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// MCP tool: omni_infer
///
/// Performs omni-directional inference with clamped parameter bounds
/// for safe, bounded reasoning.
///
/// # Security
/// - Required permission: Elevated (Level 2)
/// - Rate limit: 50 req/min per session (inference-heavy)
/// - Resource quota: 256MB memory, 30s CPU time
/// - Audit: Full logging with token usage tracking
///
/// # Parameters
/// - direction: Inference direction identifier
/// - clamped: Vector of clamped parameter values
///
/// # Returns
/// InferenceResult with outputs and confidence scores
pub async fn omni_infer(
    direction: &str,
    clamped: Vec<ClampedValue>,
) -> Result<InferenceResult, McpError> {
    // Validate direction
    if direction.is_empty() || direction.len() > 256 {
        return Err(McpError::InvalidParameter("direction must be 1-256 characters"));
    }

    // Validate all clamped values
    for cv in &clamped {
        if !cv.is_valid() {
            return Err(McpError::InvalidParameter(
                format!("clamped value '{}' outside bounds", cv.name)
            ));
        }
    }

    // Apply clamping for safety
    let safe_clamped: Vec<ClampedValue> = clamped.iter()
        .map(|cv| ClampedValue {
            name: cv.name.clone(),
            value: cv.clamped(),
            min: cv.min,
            max: cv.max,
        })
        .collect();

    // Delegate to inference engine
    let result = inference_engine::omni_infer(direction, safe_clamped).await?;

    Ok(result)
}
```

### 3.3 verify_code_node

```rust
/// Verification status for code nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStatus {
    pub node_id: Uuid,
    pub verified: bool,
    pub verification_level: VerificationLevel,
    pub checks_passed: Vec<VerificationCheck>,
    pub checks_failed: Vec<VerificationCheck>,
    pub timestamp: DateTime<Utc>,
    pub verifier_signature: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VerificationLevel {
    None = 0,
    Syntax = 1,
    Semantic = 2,
    Runtime = 3,
    Formal = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub check_name: String,
    pub check_type: CheckType,
    pub result: CheckResult,
    pub details: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CheckType {
    Syntax,
    TypeSafety,
    BoundsCheck,
    NullSafety,
    Concurrency,
    Security,
    Performance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CheckResult {
    Passed,
    Failed,
    Warning,
    Skipped,
}

/// MCP tool: verify_code_node
///
/// Performs comprehensive verification of a code node in the graph,
/// including syntax, semantic, and optional formal verification.
///
/// # Security
/// - Required permission: Admin (Level 3)
/// - Rate limit: 100 req/min per session
/// - Resource quota: 512MB memory, 60s CPU time
/// - Audit: Full verification audit trail with signatures
///
/// # Parameters
/// - node_id: UUID of the code node to verify
///
/// # Returns
/// VerificationStatus with detailed check results
pub async fn verify_code_node(node_id: Uuid) -> Result<VerificationStatus, McpError> {
    // Check node exists
    let node = code_graph::get_node(node_id).await
        .map_err(|_| McpError::NotFound("code node not found"))?;

    // Run verification pipeline
    let status = verification_pipeline::verify(&node).await?;

    // Sign verification result for non-repudiation
    let signed_status = if status.verified {
        let signature = crypto::sign_verification(&status)?;
        VerificationStatus {
            verifier_signature: Some(signature),
            ..status
        }
    } else {
        status
    };

    Ok(signed_status)
}
```

---

## 4. Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("Validation failed: {0:?}")]
    ValidationFailed(Vec<ValidationError>),

    #[error("Rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Duration },

    #[error("Unauthorized: required {required:?}, actual {actual:?}")]
    Unauthorized {
        required: PermissionLevel,
        actual: PermissionLevel,
    },

    #[error("Quota exceeded for {resource:?}")]
    QuotaExceeded {
        resource: ResourceType,
        usage: ResourceUsage,
    },

    #[error("Service unavailable")]
    ServiceUnavailable,

    #[error("Execution failed")]
    ExecutionFailed,

    #[error("Audit error: {0}")]
    AuditError(#[from] AuditError),

    #[error("Quota error: {0}")]
    QuotaError(#[from] QuotaError),
}

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(&'static str),

    #[error("Not found: {0}")]
    NotFound(&'static str),

    #[error("Internal error: {0}")]
    Internal(String),
}
```

---

## 5. Configuration Schema

```yaml
mcp_hardening:
  input_validation:
    max_input_size_bytes: 1048576  # 1MB
    max_array_items: 1000
    max_string_length: 65536
    enable_injection_detection: true
    injection_patterns:
      sql: true
      xss: true
      command: true
      path_traversal: true

  rate_limiting:
    global:
      capacity: 1000
      refill_rate_per_second: 16.67
      burst_size: 100
    session:
      capacity: 100
      refill_rate_per_second: 1.67
      burst_size: 20
    tool_overrides:
      omni_infer:
        capacity: 50
        refill_rate_per_second: 0.83
        burst_size: 10

  authentication:
    session_ttl_seconds: 3600
    refresh_token_ttl_seconds: 604800
    max_sessions_per_user: 10
    require_mfa_for_elevation: true

  audit_logging:
    target_latency_us: 100
    max_latency_us: 200
    buffer_size: 10000
    flush_interval_ms: 100
    enable_tamper_evidence: true
    retention_days: 90

  resource_quotas:
    session:
      memory_mb: 256
      cpu_time_seconds: 30
      storage_mb: 100
      concurrent_operations: 10
    user:
      memory_mb: 1024
      cpu_time_seconds_per_minute: 300
      storage_mb: 1024
      concurrent_operations: 50
    global:
      memory_mb: 16384
      cpu_time_seconds_per_minute: 3600
      storage_gb: 100
      concurrent_operations: 500
```

---

## 6. Implementation Notes

### 6.1 Performance Considerations

1. **Audit Logging**: Use async bounded channel with background writer to achieve <200us latency
2. **Rate Limiting**: Use atomic operations and lock-free data structures where possible
3. **Input Validation**: Pre-compile regex patterns at startup
4. **Resource Tracking**: Use lightweight atomic counters for concurrent operation tracking

### 6.2 Security Considerations

1. **Defense in Depth**: Each layer provides independent protection
2. **Fail Secure**: On any component failure, default to deny
3. **Audit Everything**: All security decisions must be logged
4. **Zero Trust**: Validate at every layer, never trust input

### 6.3 Testing Requirements

1. Unit tests for each component with >90% coverage
2. Integration tests for full middleware pipeline
3. Fuzz testing for input validation
4. Load testing for rate limiting accuracy
5. Penetration testing for injection prevention

---

## 7. References

- Tower middleware pattern: https://docs.rs/tower
- Token bucket algorithm: RFC 3290
- OWASP Input Validation: https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
- Tamper-evident logging: RFC 5765
