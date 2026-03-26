//! Hybrid inference engine.
//!
//! ANE for prefill + fused FFN mega-kernels.
//! Metal GPU for single-token decode.
//! CPU fallback for unsupported ops.

pub mod bench_result;
pub mod cpu;
pub mod data;
pub mod full_model;
pub mod kernels;
pub mod layer;
pub mod metal_adam;
pub mod model;
pub mod multistream;
pub mod parallel_bench;
pub mod sharding;
pub mod training;
