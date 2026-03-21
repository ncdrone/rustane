//! YaRN RoPE for DeepSeek-V2/V3 MLA attention.
//!
//! Key differences from standard RoPE:
//! 1. Frequency scaling: dims split into low/mid/high bands by beta_fast/beta_slow
//! 2. mscale: attention scale multiplier for long sequences
//! 3. Partial rotary: applies to qk_rope_head_dim (64) dims only, not full head

use crate::config::RopeScalingSection;

/// Precomputed YaRN RoPE tables.
pub struct YarnRopeTables {
    /// cos[pos * half_rope_dim + d] for d in 0..half_rope_dim
    pub cos: Vec<f32>,
    /// sin[pos * half_rope_dim + d] for d in 0..half_rope_dim
    pub sin: Vec<f32>,
    pub max_seq: usize,
    pub rope_dim: usize,
    /// Precomputed mscale for attention scaling: 0.1 * mscale_coeff * ln(factor) + 1.0
    pub mscale: f32,
}

impl YarnRopeTables {
    /// Build YaRN RoPE tables from config.
    ///
    /// `rope_dim` is qk_rope_head_dim (64 for V2-Lite/V3).
    /// `theta` is rope_theta (10000 for DeepSeek).
    pub fn build(
        max_seq: usize,
        rope_dim: usize,
        theta: f32,
        scaling: &RopeScalingSection,
    ) -> Self {
        let half = rope_dim / 2;
        let factor = scaling.factor;
        let orig_max = scaling.original_max_position_embeddings;

        // Compute per-dimension frequency with YaRN scaling
        let freqs = yarn_frequencies(rope_dim, theta, factor, orig_max,
                                     scaling.beta_fast, scaling.beta_slow);

        let mut cos = vec![0.0f32; max_seq * half];
        let mut sin = vec![0.0f32; max_seq * half];

        for pos in 0..max_seq {
            for d in 0..half {
                let angle = pos as f32 * freqs[d];
                cos[pos * half + d] = angle.cos();
                sin[pos * half + d] = angle.sin();
            }
        }

        let mscale = compute_mscale(scaling.mscale, factor);

        Self { cos, sin, max_seq, rope_dim, mscale }
    }

    /// Apply YaRN RoPE to a single rope vector in-place.
    /// `x` has length `rope_dim`. Neox-style pairing: (x[i], x[i+half]).
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.rope_dim / 2;
        debug_assert_eq!(x.len(), self.rope_dim);
        debug_assert!(pos < self.max_seq, "pos {pos} >= max_seq {}", self.max_seq);

        let base = pos * half;
        for d in 0..half {
            let cos_val = self.cos[base + d];
            let sin_val = self.sin[base + d];
            let x0 = x[d];
            let x1 = x[d + half];
            x[d] = x0 * cos_val - x1 * sin_val;
            x[d + half] = x1 * cos_val + x0 * sin_val;
        }
    }

    /// Apply YaRN RoPE to multiple heads' rope slices.
    /// `x` has layout [num_heads * rope_dim], each head gets same RoPE (shared k_pe).
    pub fn apply_all_heads(&self, x: &mut [f32], num_heads: usize, pos: usize) {
        for h in 0..num_heads {
            let start = h * self.rope_dim;
            self.apply(&mut x[start..start + self.rope_dim], pos);
        }
    }
}

/// Compute YaRN-scaled frequencies.
///
/// Three bands:
/// - High frequency (d < low_dim): no scaling (freq unchanged)
/// - Low frequency (d >= high_dim): divide by factor
/// - Mid frequency: linear interpolation between scaled and unscaled
fn yarn_frequencies(
    rope_dim: usize,
    theta: f32,
    factor: f32,
    orig_max_pos: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<f32> {
    let half = rope_dim / 2;
    let mut freqs = vec![0.0f32; half];

    // Base frequencies: 1 / theta^(2d / rope_dim)
    let base_freqs: Vec<f32> = (0..half)
        .map(|d| 1.0 / theta.powf(2.0 * d as f32 / rope_dim as f32))
        .collect();

    // Find correction range based on beta_fast / beta_slow
    let low_dim = find_correction_dim(beta_fast, rope_dim, theta, orig_max_pos);
    let high_dim = find_correction_dim(beta_slow, rope_dim, theta, orig_max_pos);

    for d in 0..half {
        let base_freq = base_freqs[d];
        let scaled_freq = base_freq / factor;

        // Linear ramp between low_dim and high_dim
        let ramp = linear_ramp_factor(d as f32, low_dim, high_dim);

        // ramp=0 → use base_freq (high freq, no scaling)
        // ramp=1 → use scaled_freq (low freq, full scaling)
        freqs[d] = base_freq * (1.0 - ramp) + scaled_freq * ramp;
    }

    freqs
}

/// Find the correction dimension for a given beta value.
/// Based on the formula from the YaRN paper.
fn find_correction_dim(beta: f32, rope_dim: usize, theta: f32, orig_max: usize) -> f32 {
    // The wavelength at dimension d is: 2π / freq(d) = 2π * theta^(2d/rope_dim)
    // We want to find d such that wavelength = orig_max * 2π / beta
    // → theta^(2d/rope_dim) = orig_max / beta
    // → d = rope_dim * ln(orig_max / beta) / (2 * ln(theta))
    let val = (orig_max as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * theta.ln()) * rope_dim as f32;
    val.max(0.0)
}

/// Linear ramp: 0 at low_dim, 1 at high_dim, clamped to [0, 1].
fn linear_ramp_factor(dim: f32, low: f32, high: f32) -> f32 {
    if high <= low {
        return if dim >= low { 1.0 } else { 0.0 };
    }
    ((dim - low) / (high - low)).clamp(0.0, 1.0)
}

/// Compute mscale from config.
/// Formula: 0.1 * mscale_coeff * ln(factor) + 1.0
pub fn compute_mscale(mscale_coeff: f32, factor: f32) -> f32 {
    if factor <= 1.0 {
        return 1.0;
    }
    0.1 * mscale_coeff * factor.ln() + 1.0
}

/// Compute the full attention scale for MLA: 1/sqrt(qk_nope_head_dim + qk_rope_head_dim) * mscale^2
/// This is the corrected formula from the research.
/// The head_dim for scaling is nope_dim (128) + rope_dim (64) = 192.
pub fn mla_attention_scale(qk_nope_head_dim: usize, qk_rope_head_dim: usize, mscale: f32) -> f32 {
    let head_dim = (qk_nope_head_dim + qk_rope_head_dim) as f32;
    (1.0 / head_dim.sqrt()) * mscale * mscale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeScalingSection;

    fn v2_lite_scaling() -> RopeScalingSection {
        RopeScalingSection {
            factor: 40.0,
            original_max_position_embeddings: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 0.707,
            mscale_all_dim: 0.707,
        }
    }

    #[test]
    fn mscale_v2_lite() {
        let ms = compute_mscale(0.707, 40.0);
        // 0.1 * 0.707 * ln(40) + 1.0 ≈ 0.0707 * 3.6889 + 1.0 ≈ 1.2608
        assert!((ms - 1.2608).abs() < 0.01, "mscale={ms}");
    }

    #[test]
    fn mscale_v3() {
        let ms = compute_mscale(1.0, 40.0);
        // 0.1 * 1.0 * ln(40) + 1.0 ≈ 1.3689
        assert!((ms - 1.3689).abs() < 0.01, "mscale={ms}");
    }

    #[test]
    fn mscale_no_scaling() {
        assert_eq!(compute_mscale(1.0, 1.0), 1.0);
        assert_eq!(compute_mscale(0.707, 0.5), 1.0);
    }

    #[test]
    fn mla_scale_factor() {
        // V2-Lite: nope=128, rope=64, mscale ≈ 1.2608
        let ms = compute_mscale(0.707, 40.0);
        let scale = mla_attention_scale(128, 64, ms);
        // 1/sqrt(192) * 1.2608^2 ≈ 0.07217 * 1.5896 ≈ 0.1147
        assert!((scale - 0.1147).abs() < 0.005, "scale={scale}");
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let scaling = v2_lite_scaling();
        let tables = YarnRopeTables::build(100, 64, 10000.0, &scaling);
        let mut x = vec![0.0f32; 64];
        x[0] = 1.0;
        x[5] = 2.0;
        x[32] = 3.0; // second half
        let orig = x.clone();
        tables.apply(&mut x, 0);
        // At pos 0: cos=1, sin=0, so x should be unchanged
        for i in 0..64 {
            assert!((x[i] - orig[i]).abs() < 1e-6, "pos=0 changed dim {i}");
        }
    }

    #[test]
    fn rope_nonzero_position_modifies() {
        let scaling = v2_lite_scaling();
        let tables = YarnRopeTables::build(200, 64, 10000.0, &scaling);
        let mut x = vec![1.0f32; 64];
        let orig = x.clone();
        tables.apply(&mut x, 100);
        // At pos 100 with non-trivial frequencies, values should change
        let max_diff: f32 = x.iter().zip(orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_diff > 0.01, "pos=100 should modify values, max_diff={max_diff}");
    }

    #[test]
    fn yarn_frequencies_high_freq_unscaled() {
        // High-frequency dims should NOT be scaled (freq unchanged from base)
        let freqs = yarn_frequencies(64, 10000.0, 40.0, 4096, 32.0, 1.0);
        let base_freq_0 = 1.0f32; // 1/10000^0 = 1.0
        // Dim 0 is the highest frequency — should be essentially unscaled
        assert!((freqs[0] - base_freq_0).abs() < 0.01,
            "high-freq dim 0 should be unscaled: got {}, expected {}", freqs[0], base_freq_0);
    }

    #[test]
    fn yarn_frequencies_low_freq_scaled() {
        // Low-frequency dims should be scaled by 1/factor
        let freqs = yarn_frequencies(64, 10000.0, 40.0, 4096, 32.0, 1.0);
        let half = 32;
        let base_freq_last = 1.0 / 10000.0f32.powf(2.0 * (half - 1) as f32 / 64.0);
        let scaled_freq_last = base_freq_last / 40.0;
        // Last dim is lowest frequency — should be close to scaled
        assert!((freqs[half - 1] - scaled_freq_last).abs() < scaled_freq_last * 0.5,
            "low-freq dim {} should be ~scaled: got {}, expected ~{}",
            half - 1, freqs[half - 1], scaled_freq_last);
    }

    #[test]
    fn apply_all_heads_consistent() {
        let scaling = v2_lite_scaling();
        let tables = YarnRopeTables::build(100, 64, 10000.0, &scaling);
        // Two heads with same values should get same RoPE
        let mut x = vec![1.0f32; 128]; // 2 heads * 64 dims
        let mut x_single = vec![1.0f32; 64];
        tables.apply_all_heads(&mut x, 2, 50);
        tables.apply(&mut x_single, 50);
        for d in 0..64 {
            assert!((x[d] - x_single[d]).abs() < 1e-6);
            assert!((x[64 + d] - x_single[d]).abs() < 1e-6);
        }
    }
}
