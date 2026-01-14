//! MCP Middleware modules.
//!
//! - `cognitive_pulse`: Real-time UTL state injection
//! - `validation`: Parameter validation with field-aware errors

mod cognitive_pulse;
pub mod validation;

pub use cognitive_pulse::CognitivePulse;
pub use validation::{
    validate_13_element_array, validate_embedder_index, validate_input, validate_optional_float,
    validate_optional_int, validate_range, validate_required_string, validate_string_length,
    validate_uuid, ValidationError,
};
