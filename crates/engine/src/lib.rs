//! Hybrid inference engine.
//!
//! ANE for prefill + fused FFN mega-kernels.
//! Metal GPU for single-token decode.
//! CPU fallback for unsupported ops.

pub mod bench_result;
pub mod checkpoint;
pub mod cpu;
pub mod data;
pub mod decode;
pub mod full_model;
pub mod inference;
pub mod kernels;
pub mod layer;
pub mod metal_adam;
pub mod metal_ffn;
pub mod metal_muon;
pub mod model;
pub mod training;
