//! CUDA acceleration for Context Graph.
//!
//! This crate provides GPU-accelerated operations for:
//! - Vector similarity search (cosine, dot product)
//! - Neural attention mechanisms
//! - Modern Hopfield network computations
//!
//! For Phase 0 (Ghost System), stub implementations run on CPU.
//! Future phases will use cudarc bindings for RTX 5090 (Blackwell) optimization.
//!
//! # Target Hardware
//!
//! - RTX 5090 (32GB GDDR7, 1.8 TB/s bandwidth)
//! - CUDA 13.1 with Compute Capability 12.0
//! - Blackwell architecture optimizations
//!
//! # Example
//!
//! ```rust,ignore
//! use context_graph_cuda::{VectorOps, StubVectorOps};
//!
//! let ops = StubVectorOps::new();
//! let similarity = ops.cosine_similarity(&vec_a, &vec_b).await?;
//! ```

pub mod error;
pub mod ops;
pub mod stub;

pub use error::{CudaError, CudaResult};
pub use ops::VectorOps;
pub use stub::StubVectorOps;
