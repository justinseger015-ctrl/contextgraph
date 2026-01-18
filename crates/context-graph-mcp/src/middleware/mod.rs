//! MCP Middleware modules.
//!
//! - `cognitive_pulse`: Real-time UTL state injection
//! - `validation`: Parameter validation with field-aware errors

mod cognitive_pulse;
pub mod validation;

pub use cognitive_pulse::CognitivePulse;
