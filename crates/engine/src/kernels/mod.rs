//! ANE kernel generators for training.
//!
//! Each generator builds an `ane::Graph` from a `ModelConfig`.
//! Pattern: placeholder → slice(acts + weights) → reshape → matmul → reshape.
//! Compile once at startup, update weights via IOSurface memcpy at runtime.

pub mod dyn_matmul;
pub mod sdpa_fwd;
pub mod sdpa_bwd;
pub mod ffn_fused;
pub mod ffn_gate_up;
pub mod ffn_down_res;
pub mod ffn_gate_proj;
pub mod ffn_up_proj;

/// ANE has a hardware dimension limit of approximately 16,384 per axis.
/// When the fused FFN kernel exceeds this, we split into smaller kernels.
pub const ANE_DIM_LIMIT: usize = 16_384;

/// FFN split level based on model dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnSplitLevel {
    /// No split needed — use single fused kernel (600M and below).
    Fused,
    /// 2-way split: gate+up combined, then down (1B–3B).
    Split2,
    /// 3-way split: gate, up, down as separate kernels (5B+).
    Split3,
}

/// Determine the required FFN split level for a model config.
pub fn ffn_split_level(cfg: &crate::model::ModelConfig) -> FfnSplitLevel {
    let fused_input_width = ffn_fused::input_spatial_width(cfg);
    let fused_output_ch = ffn_fused::output_channels(cfg);

    if fused_input_width <= ANE_DIM_LIMIT && fused_output_ch <= ANE_DIM_LIMIT {
        return FfnSplitLevel::Fused;
    }

    // Check if 2-way split fits
    let gate_up_input_width = ffn_gate_up::input_spatial_width(cfg);
    let gate_up_output_ch = ffn_gate_up::output_channels(cfg);

    if gate_up_input_width <= ANE_DIM_LIMIT && gate_up_output_ch <= ANE_DIM_LIMIT {
        return FfnSplitLevel::Split2;
    }

    // 3-way split: each projection kernel has input_width = seq + hidden
    // At 5B (hidden=8192, seq=512): 8704, well under 16384
    FfnSplitLevel::Split3
}

/// Legacy helper — returns true if any split is needed.
pub fn needs_ffn_split(cfg: &crate::model::ModelConfig) -> bool {
    ffn_split_level(cfg) != FfnSplitLevel::Fused
}
