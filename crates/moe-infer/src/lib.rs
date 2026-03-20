//! Top-level MoE inference: pipeline orchestration, KV cache, sampling.
//!
//! Coordinates the full inference pipeline:
//! 1. Token embedding
//! 2. GQA attention (RoPE + QK-norm + causal mask)
//! 3. MoE routing + expert FFN dispatch
//! 4. Output projection + sampling

pub mod attention;
pub mod config;
pub mod generate;
pub mod kv_cache;
pub mod pipeline;
pub mod rmsnorm;
pub mod sampler;
pub mod weights;
