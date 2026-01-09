# TASK-TEST-004: Security Test Suite

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-TEST-004 |
| **Title** | Security Test Suite |
| **Status** | :white_circle: todo |
| **Layer** | Testing |
| **Sequence** | 44 |
| **Estimated Days** | 2 |
| **Complexity** | Medium |

## Implements

- SEC-01 through SEC-08 verification
- OWASP testing guidelines
- Input validation testing
- PII detection accuracy testing

## Dependencies

| Task | Reason |
|------|--------|
| TASK-INTEG-011 | Security implementations to test |

## Objective

Create comprehensive security tests that verify:
1. Input validation rejects malicious input
2. PII detection catches all patterns
3. Rate limiting is enforced correctly
4. Authentication tokens expire properly
5. Authorization denies unauthorized access
6. Secrets are not leaked in logs/errors
7. Subagent isolation is enforced
8. Security events are logged correctly

## Context

Security tests are critical for verifying the constitution's security requirements (SEC-01 through SEC-08). They should cover both positive cases (valid input accepted) and negative cases (malicious input rejected).

## Scope

### In Scope

- SEC-01: Input validation tests
- SEC-02: PII detection tests
- SEC-03: Rate limiting tests
- SEC-04: Authentication tests
- SEC-05: Authorization tests
- SEC-06: Secrets management tests
- SEC-07: Isolation tests
- SEC-08: Audit logging tests

### Out of Scope

- Penetration testing
- Network security testing
- Dependency vulnerability scanning (use cargo-audit)

## Definition of Done

### SEC-01: Input Validation Tests

```rust
// tests/security/input_validation_tests.rs

use context_graph_mcp::security::input_validation::*;

#[test]
fn test_rejects_null_bytes() {
    let validator = InputValidator::new(ValidationConfig::default());

    let malicious = "hello\0world";
    let result = validator.validate_text(malicious);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ValidationError::NullByte));
}

#[test]
fn test_rejects_oversized_input() {
    let config = ValidationConfig {
        max_text_length: 100,
        ..Default::default()
    };
    let validator = InputValidator::new(config);

    let oversized = "a".repeat(101);
    let result = validator.validate_text(&oversized);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ValidationError::TooLong(_)));
}

#[test]
fn test_sanitizes_html() {
    let validator = InputValidator::new(ValidationConfig::default());

    let html_input = "<script>alert('xss')</script>Hello";
    let sanitized = validator.sanitize(html_input);

    assert!(!sanitized.contains("<script>"));
    assert!(sanitized.contains("Hello"));
}

#[test]
fn test_validates_uuid_format() {
    let validator = InputValidator::new(ValidationConfig::default());

    // Valid UUID
    assert!(validator.validate_uuid("550e8400-e29b-41d4-a716-446655440000").is_ok());

    // Invalid UUIDs
    assert!(validator.validate_uuid("not-a-uuid").is_err());
    assert!(validator.validate_uuid("").is_err());
    assert!(validator.validate_uuid("550e8400-e29b-41d4-a716").is_err()); // Truncated
}

#[test]
fn test_validates_json_depth() {
    let validator = InputValidator::new(ValidationConfig {
        max_json_depth: 10,
        ..Default::default()
    });

    // Deeply nested JSON (DoS vector)
    let deep_json = (0..20).fold("null".to_string(), |acc, _| format!("[{}]", acc));
    let json: serde_json::Value = serde_json::from_str(&deep_json).unwrap();

    let result = validator.validate_json(&json);
    assert!(result.is_err());
}

#[test]
fn test_rejects_control_characters() {
    let validator = InputValidator::new(ValidationConfig::default());

    // Bell character
    let with_bell = "hello\x07world";
    assert!(validator.validate_text(with_bell).is_err());

    // Escape sequence
    let with_escape = "hello\x1bworld";
    assert!(validator.validate_text(with_escape).is_err());
}
```

### SEC-02: PII Detection Tests

```rust
// tests/security/pii_detection_tests.rs

use context_graph_mcp::security::pii_detection::*;

#[test]
fn test_detects_ssn() {
    let detector = PiiDetector::new(PiiAction::Log);

    let text = "My SSN is 123-45-6789";
    let matches = detector.detect(text);

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::Ssn);
    assert_eq!(matches[0].matched, "123-45-6789");
}

#[test]
fn test_detects_credit_card() {
    let detector = PiiDetector::new(PiiAction::Log);

    let visa = "4111111111111111";
    let mastercard = "5500000000000004";
    let amex = "340000000000009";

    assert!(detector.contains_pii(&format!("Card: {}", visa)));
    assert!(detector.contains_pii(&format!("Card: {}", mastercard)));
    assert!(detector.contains_pii(&format!("Card: {}", amex)));
}

#[test]
fn test_detects_email() {
    let detector = PiiDetector::new(PiiAction::Log);

    let text = "Contact me at john.doe@example.com";
    let matches = detector.detect(text);

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].pii_type, PiiType::Email);
}

#[test]
fn test_detects_phone_number() {
    let detector = PiiDetector::new(PiiAction::Log);

    // US formats
    assert!(detector.contains_pii("Call me at (555) 123-4567"));
    assert!(detector.contains_pii("Phone: 555-123-4567"));
    assert!(detector.contains_pii("Tel: +1 555 123 4567"));
}

#[test]
fn test_masks_pii() {
    let detector = PiiDetector::new(PiiAction::Mask);

    let text = "My email is test@example.com and SSN is 123-45-6789";
    let masked = detector.process(text).unwrap();

    assert!(!masked.contains("test@example.com"));
    assert!(!masked.contains("123-45-6789"));
    assert!(masked.contains("[REDACTED]"));
}

#[test]
fn test_rejects_pii() {
    let detector = PiiDetector::new(PiiAction::Reject);

    let text = "My SSN is 123-45-6789";
    let result = detector.process(text);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PiiError::Rejected(_)));
}

#[test]
fn test_no_false_positives() {
    let detector = PiiDetector::new(PiiAction::Log);

    // These should NOT be detected as PII
    assert!(!detector.contains_pii("The number is 12345"));
    assert!(!detector.contains_pii("Version 1.2.3.4"));
    assert!(!detector.contains_pii("ID: ABC-123-DEF"));
}
```

### SEC-03: Rate Limiting Tests

```rust
// tests/security/rate_limit_tests.rs

use context_graph_mcp::security::rate_limiter::*;
use std::time::Duration;

#[test]
fn test_allows_within_limit() {
    let limiter = RateLimiter::new(default_rate_limits());
    let session = SessionId::new();

    // inject_context allows 100/min
    for _ in 0..100 {
        let result = limiter.check(session.clone(), "inject_context");
        assert!(result.is_ok());
        limiter.consume(session.clone(), "inject_context");
    }
}

#[test]
fn test_blocks_over_limit() {
    let limiter = RateLimiter::new(default_rate_limits());
    let session = SessionId::new();

    // consolidate_memories allows only 1/min
    let result1 = limiter.check(session.clone(), "consolidate_memories");
    assert!(result1.is_ok());
    limiter.consume(session.clone(), "consolidate_memories");

    let result2 = limiter.check(session.clone(), "consolidate_memories");
    assert!(result2.is_err());
    assert!(matches!(result2.unwrap_err(), RateLimitError::Exceeded { .. }));
}

#[test]
fn test_separate_sessions() {
    let limiter = RateLimiter::new(default_rate_limits());
    let session1 = SessionId::new();
    let session2 = SessionId::new();

    // Exhaust session1
    limiter.consume(session1.clone(), "consolidate_memories");

    // Session2 should still have quota
    let result = limiter.check(session2.clone(), "consolidate_memories");
    assert!(result.is_ok());
}

#[test]
fn test_remaining_tokens() {
    let limiter = RateLimiter::new(default_rate_limits());
    let session = SessionId::new();

    // store_memory allows 50/min
    assert_eq!(limiter.remaining(session.clone(), "store_memory"), 50);

    limiter.consume(session.clone(), "store_memory");
    assert_eq!(limiter.remaining(session.clone(), "store_memory"), 49);
}

#[tokio::test]
async fn test_refill_over_time() {
    let mut limits = HashMap::new();
    limits.insert("test".into(), RateLimit::new(1, Duration::from_millis(100)));
    let limiter = RateLimiter::new(limits);
    let session = SessionId::new();

    // Use up the limit
    limiter.consume(session.clone(), "test");
    assert!(limiter.check(session.clone(), "test").is_err());

    // Wait for refill
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Should be available again
    assert!(limiter.check(session.clone(), "test").is_ok());
}
```

### SEC-04: Authentication Tests

```rust
// tests/security/auth_tests.rs

use context_graph_mcp::security::auth::*;
use chrono::{Duration, Utc};

#[test]
fn test_session_creation() {
    let auth = SessionAuth::new(24);
    let session = auth.create_session(Permissions::default());

    assert!(session.token.len() > 0);
    assert!(session.expires_at > Utc::now());
}

#[test]
fn test_token_validation() {
    let auth = SessionAuth::new(24);
    let session = auth.create_session(Permissions::default());

    let validated = auth.validate(&session.token);
    assert!(validated.is_ok());
    assert_eq!(validated.unwrap().id, session.id);
}

#[test]
fn test_invalid_token_rejected() {
    let auth = SessionAuth::new(24);

    let fake_token = SessionToken::from("fake-token-12345");
    let result = auth.validate(&fake_token);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), AuthError::InvalidToken));
}

#[test]
fn test_expired_token_rejected() {
    let auth = SessionAuth::new(0); // 0 hour expiry = immediate
    let session = auth.create_session(Permissions::default());

    // Token already expired
    let result = auth.validate(&session.token);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), AuthError::TokenExpired));
}

#[test]
fn test_session_invalidation() {
    let auth = SessionAuth::new(24);
    let session = auth.create_session(Permissions::default());

    // Valid before invalidation
    assert!(auth.validate(&session.token).is_ok());

    // Invalidate
    auth.invalidate(&session.token);

    // Invalid after
    assert!(auth.validate(&session.token).is_err());
}

#[test]
fn test_session_refresh() {
    let auth = SessionAuth::new(1); // 1 hour
    let session = auth.create_session(Permissions::default());

    let original_expiry = session.expires_at;

    // Refresh
    let refreshed = auth.refresh(&session.token).unwrap();

    assert!(refreshed.expires_at > original_expiry);
}
```

### SEC-05: Authorization Tests

```rust
// tests/security/authz_tests.rs

use context_graph_mcp::security::authz::*;

#[test]
fn test_read_permission_for_search() {
    let authz = Authorizer::new();
    let session = Session {
        permissions: Permissions::new().with(Permission::Read),
        ..Default::default()
    };

    let result = authz.authorize(&session, "search_graph");
    assert!(result.is_ok());
}

#[test]
fn test_write_permission_for_store() {
    let authz = Authorizer::new();

    // Read-only session
    let read_session = Session {
        permissions: Permissions::new().with(Permission::Read),
        ..Default::default()
    };

    // Should fail
    let result = authz.authorize(&read_session, "store_memory");
    assert!(result.is_err());

    // Write session
    let write_session = Session {
        permissions: Permissions::new().with(Permission::Write),
        ..Default::default()
    };

    // Should succeed
    assert!(authz.authorize(&write_session, "store_memory").is_ok());
}

#[test]
fn test_admin_permission_for_consolidate() {
    let authz = Authorizer::new();

    // Non-admin session
    let user_session = Session {
        permissions: Permissions::new()
            .with(Permission::Read)
            .with(Permission::Write),
        ..Default::default()
    };

    // Should fail
    let result = authz.authorize(&user_session, "consolidate_memories");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), AuthzError::PermissionDenied { .. }));
}
```

### SEC-06 & SEC-07 & SEC-08 Tests

```rust
// tests/security/misc_security_tests.rs

// SEC-06: Secrets Management
#[test]
fn test_secrets_from_env() {
    std::env::set_var("CONTEXT_GRAPH_DB_PATH", "/tmp/test.db");
    let path = SecretsManager::db_path().unwrap();
    assert_eq!(path, PathBuf::from("/tmp/test.db"));
}

#[test]
fn test_missing_secrets_error() {
    std::env::remove_var("CONTEXT_GRAPH_DB_PATH");
    let result = SecretsManager::db_path();
    assert!(result.is_err());
}

// SEC-07: Isolation
#[test]
fn test_subagent_isolation() {
    let manager = IsolationManager::new();

    let parent = SessionId::new();
    let scope = manager.create_scope(parent.clone());

    // Subagent can only access its own namespace
    assert!(scope.allowed_namespaces.contains(&parent.to_string()));
    assert!(!scope.cross_session_access);
}

#[test]
fn test_cross_session_requires_grant() {
    let manager = IsolationManager::new();
    let parent = SessionId::new();
    let mut scope = manager.create_scope(parent);

    let other_memory = Uuid::new_v4();

    // Cannot access other session's memory
    assert!(!manager.check_access(&scope, other_memory));

    // Grant cross-session
    manager.grant_cross_session(&mut scope);

    // Now can access (in theory)
    assert!(scope.cross_session_access);
}

// SEC-08: Audit Logging
#[test]
fn test_security_event_logging() {
    let log_capture = Arc::new(RwLock::new(Vec::new()));

    // Configure tracing to capture logs
    // (simplified - real impl would use tracing-subscriber)

    SecurityAuditLog::log(SecurityEvent::AuthenticationFailure {
        session_token: "fake".into(),
        reason: "Invalid token".into(),
    });

    SecurityAuditLog::log(SecurityEvent::RateLimitExceeded {
        session_id: "session-1".into(),
        tool: "consolidate_memories".into(),
        limit: 1,
    });

    // Verify logs are structured JSON
    // (would check captured logs in real implementation)
}
```

### Constraints

| Constraint | Target |
|------------|--------|
| Security test coverage | > 90% of SEC-* |
| False positive rate | < 1% |
| False negative rate | < 0.1% |

## Verification

- [ ] All SEC-01 input validation tests pass
- [ ] All SEC-02 PII patterns detected with < 1% FP
- [ ] All SEC-03 rate limits enforced correctly
- [ ] All SEC-04 auth scenarios covered
- [ ] All SEC-05 authz scenarios covered
- [ ] SEC-06 secrets not leaked
- [ ] SEC-07 isolation enforced
- [ ] SEC-08 events logged

## Files to Create

| File | Purpose |
|------|---------|
| `tests/security/mod.rs` | Security test module |
| `tests/security/input_validation_tests.rs` | SEC-01 tests |
| `tests/security/pii_detection_tests.rs` | SEC-02 tests |
| `tests/security/rate_limit_tests.rs` | SEC-03 tests |
| `tests/security/auth_tests.rs` | SEC-04 tests |
| `tests/security/authz_tests.rs` | SEC-05 tests |
| `tests/security/misc_security_tests.rs` | SEC-06/07/08 tests |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missing edge cases | Medium | High | Use fuzzing (TEST-002) |
| Regex bypass | Low | Critical | Multiple pattern sources |
| Test maintenance | Medium | Low | Good abstractions |

## Traceability

- Source: Constitution SEC-01 through SEC-08
- Related: TASK-INTEG-011 (Security Implementation)
