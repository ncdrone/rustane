//! GQA attention for Qwen3-MoE-30B inference.
//!
//! Key facts:
//! - 32 Q heads, 4 KV heads (GQA ratio 8:1)
//! - head_dim = 128
//! - RoPE: theta=1e6, neox-style pairing (i, i + head_dim/2)
//! - QK-norm: per-head RMSNorm on Q and K before attention scores
//! - Causal mask: position s attends to t where t <= s (not t < s)

use half::f16;

use crate::kv_cache::KvCache;
use crate::rmsnorm::per_head_rmsnorm;
use crate::weights::LayerWeights;

/// GQA configuration.
#[derive(Clone, Debug)]
pub struct GqaConfig {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl GqaConfig {
    /// GQA group size: how many Q heads share one KV head.
    pub fn group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
}

// ---------- RoPE ----------

/// Precomputed RoPE cos/sin tables.
/// Layout: [max_seq * half_dim] where half_dim = head_dim / 2.
/// For position p and dimension d: index = p * half_dim + d.
pub struct RopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub max_seq: usize,
    pub head_dim: usize,
}

impl RopeTables {
    /// Build RoPE tables with given theta.
    /// Neox-style: freq[d] = 1.0 / theta^(2d / head_dim) for d in 0..head_dim/2
    pub fn build(max_seq: usize, head_dim: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq * half];
        let mut sin = vec![0.0f32; max_seq * half];

        for pos in 0..max_seq {
            for d in 0..half {
                let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half + d] = angle.cos();
                sin[pos * half + d] = angle.sin();
            }
        }

        Self { cos, sin, max_seq, head_dim }
    }

    /// Apply RoPE to a single head vector in-place.
    /// Neox-style pairing: pairs (x[i], x[i + half]) for i in 0..half.
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        debug_assert_eq!(x.len(), self.head_dim);
        debug_assert!(pos < self.max_seq);

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

    /// Apply RoPE to all heads in a flat [num_heads * head_dim] vector.
    pub fn apply_all_heads(&self, x: &mut [f32], num_heads: usize, pos: usize) {
        for h in 0..num_heads {
            let start = h * self.head_dim;
            self.apply(&mut x[start..start + self.head_dim], pos);
        }
    }
}

// ---------- GQA Forward ----------

/// Matrix-vector multiply: y = W @ x using Accelerate BLAS.
/// W is [out, in] row-major as f16, x is [in] as f32.
fn matvec_f16(w: &[f16], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    crate::blas::sgemv_f16(w, x, &mut y, out_dim, in_dim);
    y
}

/// Single-head attention: Q @ K^T / sqrt(d) + causal_mask → softmax → @ V
/// q: [head_dim], k_cache: [seq_len * head_dim], v_cache: [seq_len * head_dim]
/// Returns: [head_dim]
fn single_head_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
    _current_pos: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q @ K^T / sqrt(d) — scores for each cached position
    let mut scores = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let k_t = &k_cache[t * head_dim..(t + 1) * head_dim];
        let mut dot = 0.0f64;
        for d in 0..head_dim {
            dot += q[d] as f64 * k_t[d] as f64;
        }
        scores[t] = dot as f32 * scale;
        // Causal mask: current position can attend to all positions <= current_pos.
        // Since we only store up to seq_len positions and seq_len <= current_pos + 1,
        // all cached positions are valid (no masking needed for single-token decode).
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in &mut exps {
        *e /= sum;
    }

    // Weighted sum of V
    let mut out = vec![0.0f32; head_dim];
    for t in 0..seq_len {
        let v_t = &v_cache[t * head_dim..(t + 1) * head_dim];
        let w = exps[t];
        for d in 0..head_dim {
            out[d] += w * v_t[d];
        }
    }
    out
}

/// Full GQA attention forward pass for a single token.
///
/// Steps:
/// 1. Q = x @ q_proj → [num_q_heads * head_dim]
/// 2. K = x @ k_proj → [num_kv_heads * head_dim]
/// 3. V = x @ v_proj → [num_kv_heads * head_dim]
/// 4. QK-norm: per-head RMSNorm on Q (using q_norm) and K (using k_norm)
/// 5. RoPE on Q and K
/// 6. Store K, V in cache
/// 7. Per Q head: attend to corresponding KV head (GQA broadcast)
/// 8. Concat heads → o_proj → output [hidden_size]
pub fn gqa_forward(
    x: &[f32],
    lw: &LayerWeights<'_>,
    cache: &mut KvCache,
    layer: usize,
    pos: usize,
    rope: &RopeTables,
    cfg: &GqaConfig,
    eps: f32,
) -> Vec<f32> {
    let hidden = x.len();
    let q_dim = cfg.num_q_heads * cfg.head_dim;
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;

    // 1. Q/K/V projections
    let mut q = matvec_f16(lw.q_proj, x, q_dim, hidden);
    let mut k = matvec_f16(lw.k_proj, x, kv_dim, hidden);
    let v = matvec_f16(lw.v_proj, x, kv_dim, hidden);

    // 2. QK-norm: per-head RMSNorm
    per_head_rmsnorm(&mut q, &f32_from_f32_slice(lw.q_norm), cfg.num_q_heads, eps);
    per_head_rmsnorm(&mut k, &f32_from_f32_slice(lw.k_norm), cfg.num_kv_heads, eps);

    // 3. RoPE
    rope.apply_all_heads(&mut q, cfg.num_q_heads, pos);
    rope.apply_all_heads(&mut k, cfg.num_kv_heads, pos);

    // 4. Store K, V in cache
    cache.store(layer, pos, &k, &v);
    let seq_len = pos + 1;

    // 5. Per Q head: attend to corresponding KV head
    let group_size = cfg.group_size();
    let mut concat = vec![0.0f32; q_dim];

    for qh in 0..cfg.num_q_heads {
        let kv_h = qh / group_size; // GQA broadcast (integer division)
        let q_head = &q[qh * cfg.head_dim..(qh + 1) * cfg.head_dim];
        let k_cache = cache.get_k(layer, kv_h, seq_len);
        let v_cache = cache.get_v(layer, kv_h, seq_len);

        let head_out = single_head_attention(q_head, k_cache, v_cache, cfg.head_dim, seq_len, pos);
        concat[qh * cfg.head_dim..(qh + 1) * cfg.head_dim].copy_from_slice(&head_out);
    }

    // 6. Output projection
    matvec_f16(lw.o_proj, &concat, hidden, q_dim)
}

/// Helper: q_norm and k_norm are already f32 from weights.rs, just copy the slice ref.
/// This is a no-op passthrough — the function exists for API clarity.
#[inline]
fn f32_from_f32_slice(s: &[f32]) -> Vec<f32> {
    s.to_vec()
}

/// Prefill: process multiple input tokens at once (sequential, not parallel).
/// Returns the hidden state after the last token.
pub fn gqa_prefill(
    tokens_hidden: &[Vec<f32>],
    lw: &LayerWeights<'_>,
    cache: &mut KvCache,
    layer: usize,
    start_pos: usize,
    rope: &RopeTables,
    cfg: &GqaConfig,
    eps: f32,
) -> Vec<Vec<f32>> {
    tokens_hidden
        .iter()
        .enumerate()
        .map(|(i, x)| {
            gqa_forward(x, lw, cache, layer, start_pos + i, rope, cfg, eps)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_position_zero() {
        let rope = RopeTables::build(4096, 128, 1_000_000.0);
        // At pos 0: cos should be 1.0, sin should be 0.0
        assert!((rope.cos[0] - 1.0).abs() < 1e-6);
        assert!(rope.sin[0].abs() < 1e-6);
        // All dimensions at pos 0
        for d in 0..64 {
            assert!((rope.cos[d] - 1.0).abs() < 1e-6, "cos[0][{d}] != 1.0");
            assert!(rope.sin[d].abs() < 1e-6, "sin[0][{d}] != 0.0");
        }
    }

    #[test]
    fn rope_theta_1e6() {
        let rope = RopeTables::build(4096, 128, 1_000_000.0);
        let half = 64;

        // High-freq dim (d=0): freq = 1.0, angle at pos 512 = 512.0
        let idx = 512 * half;
        assert!((rope.cos[idx] - (512.0f32).cos()).abs() < 1e-4);

        // Low-freq dim (d=63): freq = 1/1e6^(126/128) ≈ very small
        // angle ≈ 0, so cos ≈ 1.0
        let idx_low = 512 * half + 63;
        assert!(rope.cos[idx_low] > 0.999, "low-freq dim should barely rotate");
    }

    #[test]
    fn rope_neox_pairing() {
        let rope = RopeTables::build(100, 8, 10000.0);
        // Verify neox pairing: (x[0], x[4]) are paired, NOT (x[0], x[1])
        let mut x = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        rope.apply(&mut x, 1);
        // After RoPE at pos 1: x[0] and x[4] should be modified (paired)
        // x[0] = 1.0 * cos - 0.0 * sin = cos(freq_0)
        // x[4] = 0.0 * cos + 1.0 * sin = sin(freq_0)
        let freq_0 = 1.0f32; // 1/10000^0 = 1
        assert!((x[0] - (1.0f32).cos()).abs() < 1e-5);
        assert!((x[4] - (1.0f32).sin()).abs() < 1e-5);
    }

    #[test]
    fn single_head_attention_basic() {
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k_cache = vec![
            1.0, 0.0, 0.0, 0.0, // pos 0: aligned with q
            0.0, 1.0, 0.0, 0.0, // pos 1: orthogonal
        ];
        let v_cache = vec![
            10.0, 20.0, 30.0, 40.0, // pos 0
            50.0, 60.0, 70.0, 80.0, // pos 1
        ];
        let out = single_head_attention(&q, &k_cache, &v_cache, head_dim, 2, 1);

        // Q aligns with K[0], so attention should weight pos 0 more heavily
        // score[0] = 1.0/sqrt(4) = 0.5, score[1] = 0.0
        // softmax: [exp(0.5)/Z, exp(0)/Z] where Z = exp(0.5) + exp(0)
        let s0 = 0.5f32.exp();
        let s1 = 1.0f32;
        let z = s0 + s1;
        let w0 = s0 / z;
        let w1 = s1 / z;
        assert!((out[0] - (w0 * 10.0 + w1 * 50.0)).abs() < 1e-4);
    }
}
