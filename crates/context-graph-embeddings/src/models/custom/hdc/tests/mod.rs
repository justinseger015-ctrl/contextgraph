//! Tests for the HDC (Hyperdimensional Computing) model.
//!
//! Tests are organized into modules by functionality:
//! - construction: Model creation and initialization
//! - hypervector: Random hypervector generation
//! - operations: Bind, bundle, permute, and similarity
//! - encoding: Text encoding and projection
//! - embedding: EmbeddingModel trait implementation
//! - edge_cases: Special scenarios and thread safety

use super::*;
use crate::error::EmbeddingError;
use crate::traits::EmbeddingModel;
use crate::types::{InputType, ModelId, ModelInput};

mod construction;
mod edge_cases;
mod embedding;
mod encoding;
mod hypervector;
mod operations;
