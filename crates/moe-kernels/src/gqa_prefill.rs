//! ANE fused GQA prefill graph for Qwen3-MoE-30B.
//!
//! Processes ALL prefill tokens in one ANE dispatch per layer:
//! QKV projection → QK-norm → neox RoPE → grouped causal SDPA.
//!
//! Input: [1, hidden, 1, pad16(seq + q_dim + kv_dim + kv_dim)]
//!   spatial[0:seq]                     = token activations
//!   spatial[seq:seq+q_dim]             = Wq [hidden, q_dim] (transposed from CPU [q_dim, hidden])
//!   spatial[seq+q_dim:..+kv_dim]       = Wk [hidden, kv_dim]
//!   spatial[..+kv_dim:..+2*kv_dim]     = Wv [hidden, kv_dim]
//!
//! Output: [1, q_dim+kv_dim+kv_dim, 1, seq]
//!   channels[0:q_dim]                  = attention output (before O_proj)
//!   channels[q_dim:q_dim+kv_dim]       = K after QK-norm + RoPE (for KV cache)
//!   channels[q_dim+kv_dim:..]          = V raw projection (for KV cache)
//!
//! Adapted from engine/kernels/sdpa_fwd.rs with:
//! - Neox-style RoPE (not interleaved pairs)
//! - QK-norm (per-head RMSNorm with pow(-0.5), NOT rsqrt)
//! - GQA 8:1 via grouped attention (no K/V expansion needed)

use ane_bridge::ane::{Graph, Shape};

/// Neox-style RoPE cos/sin tables.
/// cos[p * hd + d] = cos(p * freq[d % half]), same pattern for sin.
fn neox_rope_table(seq: usize, hd: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let mut cos_buf = vec![0.0f32; seq * hd];
    let mut sin_buf = vec![0.0f32; seq * hd];
    for p in 0..seq {
        for d in 0..hd {
            let freq_idx = d % half;
            let freq = 1.0 / theta.powf(2.0 * freq_idx as f32 / hd as f32);
            let angle = p as f32 * freq;
            cos_buf[p * hd + d] = angle.cos();
            sin_buf[p * hd + d] = angle.sin();
        }
    }
    (cos_buf, sin_buf)
}

/// Causal mask: 0 where t2 <= t, -65504 (fp16 -inf) where t2 > t.
fn causal_mask(seq: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq * seq];
    for t in 0..seq {
        for t2 in 0..seq {
            if t2 > t {
                mask[t * seq + t2] = -65504.0;
            }
        }
    }
    mask
}

/// Pad to next multiple of 16.
pub fn pad16(x: usize) -> usize {
    (x + 15) & !15
}

/// Weight spatial width: q_dim + kv_dim + kv_dim.
pub fn weight_spatial_width(q_dim: usize, kv_dim: usize) -> usize {
    q_dim + kv_dim + kv_dim
}

/// Output channel count: q_dim + kv_dim + kv_dim.
pub fn output_channels(q_dim: usize, kv_dim: usize) -> usize {
    q_dim + kv_dim + kv_dim
}

/// Build the fused GQA prefill graph with two inputs:
/// - Input 0 (acts): [1, hidden, 1, seq] — token activations (re-staged per layer)
/// - Input 1 (wts):  [1, hidden, 1, q_dim+kv_dim+kv_dim] — weights (pre-staged at load time)
///
/// Output: [1, q_dim+kv_dim+kv_dim, 1, seq]
pub fn build_gqa_prefill(
    hidden: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq: usize,
    rope_theta: f32,
    eps: f32,
    q_norm_w: &[f32],
    k_norm_w: &[f32],
) -> Graph {
    assert!(seq >= 64, "ANE requires spatial width >= 64");
    assert!(seq % 16 == 0, "seq must be multiple of 16");
    assert_eq!(q_norm_w.len(), head_dim);
    assert_eq!(k_norm_w.len(), head_dim);

    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let gqa_group = num_q_heads / num_kv_heads;
    let half_hd = head_dim / 2;
    let wts_sp = weight_spatial_width(q_dim, kv_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut g = Graph::new();

    // ── Two input placeholders ──
    // Input 0: activations [1, hidden, 1, seq] — re-staged per layer (tiny)
    let xn = g.placeholder(Shape { batch: 1, channels: hidden, height: 1, width: seq });
    // Input 1: weights [1, hidden, 1, wts_sp] — pre-staged once at load time
    let wts = g.placeholder(Shape { batch: 1, channels: hidden, height: 1, width: wts_sp });

    // ── Slice weights from spatial dimension ──
    let wq = g.slice(wts, [0, 0, 0, 0], [1, hidden, 1, q_dim]);
    let wk = g.slice(wts, [0, 0, 0, q_dim], [1, hidden, 1, kv_dim]);
    let wv = g.slice(wts, [0, 0, 0, q_dim + kv_dim], [1, hidden, 1, kv_dim]);

    // ── QKV projection: X @ W ──
    // Reshape xn [1,hidden,1,seq] → [1,1,hidden,seq] → transpose → [1,1,seq,hidden]
    let xn2 = g.reshape(xn, Shape { batch: 1, channels: 1, height: hidden, width: seq });
    let xnt = g.transpose(xn2, [0, 1, 3, 2]);

    let wq2 = g.reshape(wq, Shape { batch: 1, channels: 1, height: hidden, width: q_dim });
    let wk2 = g.reshape(wk, Shape { batch: 1, channels: 1, height: hidden, width: kv_dim });
    let wv2 = g.reshape(wv, Shape { batch: 1, channels: 1, height: hidden, width: kv_dim });

    // [1,1,seq,hidden] @ [1,1,hidden,X] → [1,1,seq,X]
    let qm = g.matrix_multiplication(xnt, wq2, false, false);
    let km = g.matrix_multiplication(xnt, wk2, false, false);
    let vm = g.matrix_multiplication(xnt, wv2, false, false);

    // ── Reshape to heads: [1, heads, seq, hd] ──
    let qt = g.transpose(qm, [0, 1, 3, 2]); // [1,1,q_dim,seq]
    let qf = g.reshape(qt, Shape { batch: 1, channels: q_dim, height: 1, width: seq });
    let q4 = g.reshape(qf, Shape { batch: 1, channels: num_q_heads, height: head_dim, width: seq });
    let q = g.transpose(q4, [0, 1, 3, 2]); // [1,32,seq,128]

    let kt = g.transpose(km, [0, 1, 3, 2]);
    let kf = g.reshape(kt, Shape { batch: 1, channels: kv_dim, height: 1, width: seq });
    let k4 = g.reshape(kf, Shape { batch: 1, channels: num_kv_heads, height: head_dim, width: seq });
    let k = g.transpose(k4, [0, 1, 3, 2]); // [1,4,seq,128]

    let vt = g.transpose(vm, [0, 1, 3, 2]);
    let vf = g.reshape(vt, Shape { batch: 1, channels: kv_dim, height: 1, width: seq });
    let v4 = g.reshape(vf, Shape { batch: 1, channels: num_kv_heads, height: head_dim, width: seq });
    let v = g.transpose(v4, [0, 1, 3, 2]); // [1,4,seq,128]

    // ── QK-norm: per-head RMSNorm ──
    // CRITICAL: pow(-0.5), NOT rsqrt — ANE compiler fails on rsqrt after reduce ops.
    // CRITICAL: add epsilon BEFORE pow — zero-magnitude vectors produce inf in fp16.
    let eps_const = g.constant_with_scalar(eps, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let neg_half = g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });

    // Q norm: reduce_mean(Q², axis=width) → add(eps) → pow(-0.5) → scale
    let q_sq = g.multiplication(q, q);
    let q_mean = g.reduce_mean(q_sq, 3); // [1,32,seq,1]
    let q_me = g.addition(q_mean, eps_const);
    let q_rms = g.power(q_me, neg_half); // [1,32,seq,1]
    let q_normed = g.multiplication(q, q_rms); // broadcasts width 1→128
    let q_nw = g.constant(q_norm_w, Shape { batch: 1, channels: 1, height: 1, width: head_dim });
    let q_scaled = g.multiplication(q_normed, q_nw);

    // K norm: same pattern
    let k_sq = g.multiplication(k, k);
    let k_mean = g.reduce_mean(k_sq, 3); // [1,4,seq,1]
    let k_me = g.addition(k_mean, eps_const);
    let k_rms = g.power(k_me, neg_half);
    let k_normed = g.multiplication(k, k_rms);
    let k_nw = g.constant(k_norm_w, Shape { batch: 1, channels: 1, height: 1, width: head_dim });
    let k_scaled = g.multiplication(k_normed, k_nw);

    // ── Neox RoPE ──
    // rotated = concat(-x[half:], x[:half]) then result = x * cos + rotated * sin
    let (cos_data, sin_data) = neox_rope_table(seq, head_dim, rope_theta);
    let rope_cos = g.constant(&cos_data, Shape { batch: 1, channels: 1, height: seq, width: head_dim });
    let rope_sin = g.constant(&sin_data, Shape { batch: 1, channels: 1, height: seq, width: head_dim });
    let neg1 = g.constant_with_scalar(-1.0, Shape { batch: 1, channels: 1, height: 1, width: 1 });

    // Q RoPE
    let q_first = g.slice(q_scaled, [0, 0, 0, 0], [1, num_q_heads, seq, half_hd]);
    let q_second = g.slice(q_scaled, [0, 0, 0, half_hd], [1, num_q_heads, seq, half_hd]);
    let q_neg_second = g.multiplication(q_second, neg1);
    let q_rotated = g.concat(&[q_neg_second, q_first], 3);
    let q_cos = g.multiplication(q_scaled, rope_cos);
    let q_sin = g.multiplication(q_rotated, rope_sin);
    let q_rope = g.addition(q_cos, q_sin); // [1,32,seq,128]

    // K RoPE
    let k_first = g.slice(k_scaled, [0, 0, 0, 0], [1, num_kv_heads, seq, half_hd]);
    let k_second = g.slice(k_scaled, [0, 0, 0, half_hd], [1, num_kv_heads, seq, half_hd]);
    let k_neg_second = g.multiplication(k_second, neg1);
    let k_rotated = g.concat(&[k_neg_second, k_first], 3);
    let k_cos = g.multiplication(k_scaled, rope_cos);
    let k_sin = g.multiplication(k_rotated, rope_sin);
    let k_rope = g.addition(k_cos, k_sin); // [1,4,seq,128]

    // ── Grouped causal attention ──
    // For each KV head, compute attention for its Q group (8 heads).
    // Uses matmul broadcast: [1,8,seq,hd] @ [1,1,hd,seq] → [1,8,seq,seq]
    let scale_const = g.constant_with_scalar(scale, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let mask_data = causal_mask(seq);
    let mask_const = g.constant(&mask_data, Shape { batch: 1, channels: 1, height: seq, width: seq });

    let mut attn_groups = Vec::with_capacity(num_kv_heads);
    for h in 0..num_kv_heads {
        let q_group = g.slice(q_rope, [0, h * gqa_group, 0, 0], [1, gqa_group, seq, head_dim]);
        let k_h = g.slice(k_rope, [0, h, 0, 0], [1, 1, seq, head_dim]);
        let v_h = g.slice(v, [0, h, 0, 0], [1, 1, seq, head_dim]);

        // scores = Q @ K^T * scale + mask → softmax → @ V
        let scores = g.matrix_multiplication(q_group, k_h, false, true);
        let scores_s = g.multiplication(scores, scale_const);
        let scores_m = g.addition(scores_s, mask_const);
        let attn_w = g.soft_max(scores_m, -1);
        let attn_h = g.matrix_multiplication(attn_w, v_h, false, false);
        attn_groups.push(attn_h);
    }
    let attn_out = g.concat(&attn_groups, 1); // [1,32,seq,128]

    // ── Output: concat(attn_out, K_rope, V) → [1, 5120, 1, seq] ──
    let ao_t = g.transpose(attn_out, [0, 1, 3, 2]); // [1,32,128,seq]
    let ao_f = g.reshape(ao_t, Shape { batch: 1, channels: q_dim, height: 1, width: seq });

    let kr_t = g.transpose(k_rope, [0, 1, 3, 2]); // [1,4,128,seq]
    let kr_f = g.reshape(kr_t, Shape { batch: 1, channels: kv_dim, height: 1, width: seq });

    let v_t = g.transpose(v, [0, 1, 3, 2]); // [1,4,128,seq]
    let v_f = g.reshape(v_t, Shape { batch: 1, channels: kv_dim, height: 1, width: seq });

    let _output = g.concat(&[ao_f, kr_f, v_f], 1);

    g
}
