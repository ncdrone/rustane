//! Top-level MoE inference: pipeline orchestration, KV cache, sampling.
//!
//! Coordinates the full inference pipeline:
//! 1. Token embedding
//! 2. GQA attention (RoPE + QK-norm + causal mask)
//! 3. MoE routing + expert FFN dispatch
//! 4. Output projection + sampling

pub mod ane_mla;
pub mod attention;
pub mod blas;
pub mod config;
pub mod fp8;
pub mod generate;
pub mod generate_v2;
pub mod kv_cache;
pub mod mla_attention;
pub mod pipeline;
pub mod rmsnorm;
pub mod sampler;
pub mod weights;
pub mod yarn_rope;
