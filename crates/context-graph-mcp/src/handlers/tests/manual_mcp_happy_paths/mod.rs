//! Manual MCP Happy Path Testing
//!
//! This test module manually tests ALL MCP handlers to verify values show up correctly.
//! Uses RocksDB with tempdir for isolated testing.
//!
//! Run with: cargo test -p context-graph-mcp manual_mcp_happy_paths -- --nocapture

mod common;

mod memory_tests;
mod search_tests;
mod purpose_utl_tests;
mod system_tests;
mod johari_tests;
mod tools_init_tests;
mod gwt_consciousness_tests;
mod comprehensive_summary;
mod stubfix_tests;
