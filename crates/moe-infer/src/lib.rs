//! Top-level MoE inference: pipeline orchestration, KV cache, sampling.
//!
//! Coordinates the full inference pipeline:
//! 1. Token embedding
//! 2. Attention (MLA with compressed KV cache)
//! 3. MoE routing + expert FFN dispatch
//! 4. Output projection + sampling

pub mod config;
pub mod pipeline;
pub mod sampler;
