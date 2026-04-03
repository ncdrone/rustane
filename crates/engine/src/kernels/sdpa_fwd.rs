//! SDPA Forward kernel: QKV projection + RoPE + attention + output.
//!
//! Input IOSurface: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM]
//!   sp[0:SEQ]                     = xnorm [DIM, SEQ]
//!   sp[SEQ:SEQ+Q_DIM]             = Wq [DIM, Q_DIM]
//!   sp[SEQ+Q_DIM:SEQ+Q_DIM+KVD]  = Wk [DIM, KV_DIM]
//!   sp[SEQ+Q_DIM+KVD:...]         = Wv [DIM, KV_DIM]
//!
//! Output: [1, Q_DIM+Q_DIM+KV_DIM+KV_DIM+DIM, 1, SEQ]
//!   = concat(attn_out, Q_rope, K_rope, V, xnorm_passthrough)
//!
//! For gpt_karpathy (MHA): Q_DIM = KV_DIM = DIM = 768, HEADS = KV_HEADS = 6

use crate::model::ModelConfig;
use ane_bridge::ane::{Graph, Shape, Tensor};

/// Generate RoPE cos/sin tables as f32 data for BLOBFILE constants.
fn rope_table(seq: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos_buf = vec![0.0f32; seq * hd];
    let mut sin_buf = vec![0.0f32; seq * hd];
    for p in 0..seq {
        for i in 0..hd / 2 {
            let theta = p as f32 / 10000.0f32.powf(2.0 * i as f32 / hd as f32);
            let c = theta.cos();
            let s = theta.sin();
            cos_buf[p * hd + 2 * i] = c;
            cos_buf[p * hd + 2 * i + 1] = c;
            sin_buf[p * hd + 2 * i] = s;
            sin_buf[p * hd + 2 * i + 1] = s;
        }
    }
    (cos_buf, sin_buf)
}

/// Generate causal mask: 0 where t2 <= t, -65504 (fp16 -inf) where t2 > t.
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

fn tile_kv_heads(
    g: &mut Graph,
    kv: Tensor,
    kv_heads: usize,
    gqa_ratio: usize,
    seq: usize,
    hd: usize,
) -> Tensor {
    if gqa_ratio == 1 {
        return kv;
    }
    let mut tiled = Vec::with_capacity(kv_heads * gqa_ratio);
    for kv_head in 0..kv_heads {
        let head = g.slice(kv, [0, kv_head, 0, 0], [1, 1, seq, hd]);
        for _ in 0..gqa_ratio {
            tiled.push(head);
        }
    }
    g.concat(&tiled, 1)
}

fn build_impl(cfg: &ModelConfig, split_inputs: bool) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let heads = cfg.heads;
    let kv_heads = cfg.kv_heads;
    let hd = cfg.hd;
    let gqa_ratio = cfg.gqa_ratio;

    let sp_in = seq + q_dim + kv_dim + kv_dim;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut g = Graph::new();
    let input = if split_inputs {
        let xn = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let wq = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: q_dim,
        });
        let wk = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: kv_dim,
        });
        let wv = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: kv_dim,
        });
        g.concat(&[xn, wq, wk, wv], 3)
    } else {
        g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: sp_in,
        })
    };

    // ── Slice inputs from spatial dimension ──
    let xn = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
    let wq = g.slice(input, [0, 0, 0, seq], [1, dim, 1, q_dim]);
    let wk = g.slice(input, [0, 0, 0, seq + q_dim], [1, dim, 1, kv_dim]);
    let wv = g.slice(input, [0, 0, 0, seq + q_dim + kv_dim], [1, dim, 1, kv_dim]);

    // ── QKV projection: xnorm @ W ──
    // Reshape xnorm: [1,DIM,1,SEQ] → [1,1,DIM,SEQ] → transpose → [1,1,SEQ,DIM]
    let xn2 = g.reshape(
        xn,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: seq,
        },
    );
    let xnt = g.transpose(xn2, [0, 1, 3, 2]); // [1,1,SEQ,DIM]

    // Reshape weights to [1,1,DIM,X]
    let wq2 = g.reshape(
        wq,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: q_dim,
        },
    );
    let wk2 = g.reshape(
        wk,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: kv_dim,
        },
    );
    let wv2 = g.reshape(
        wv,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: kv_dim,
        },
    );

    // QKV matmul: [1,1,SEQ,DIM] @ [1,1,DIM,X] → [1,1,SEQ,X]
    let qm = g.matrix_multiplication(xnt, wq2, false, false);
    let km = g.matrix_multiplication(xnt, wk2, false, false);
    let vm = g.matrix_multiplication(xnt, wv2, false, false);

    // Transpose back and reshape to [1,X,1,SEQ]
    let qt = g.transpose(qm, [0, 1, 3, 2]);
    let kt = g.transpose(km, [0, 1, 3, 2]);
    let vt = g.transpose(vm, [0, 1, 3, 2]);
    let qf = g.reshape(
        qt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );
    let kf = g.reshape(
        kt,
        Shape {
            batch: 1,
            channels: kv_dim,
            height: 1,
            width: seq,
        },
    );
    let vf = g.reshape(
        vt,
        Shape {
            batch: 1,
            channels: kv_dim,
            height: 1,
            width: seq,
        },
    );

    // ── Reshape to heads: [1,X,1,SEQ] → [1,HEADS,HD,SEQ] → [1,HEADS,SEQ,HD] ──
    let q4 = g.reshape(
        qf,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let q = g.transpose(q4, [0, 1, 3, 2]); // [1,HEADS,SEQ,HD]
    let k4 = g.reshape(
        kf,
        Shape {
            batch: 1,
            channels: kv_heads,
            height: hd,
            width: seq,
        },
    );
    let k_base = g.transpose(k4, [0, 1, 3, 2]);
    let k = tile_kv_heads(&mut g, k_base, kv_heads, gqa_ratio, seq, hd);
    let v4 = g.reshape(
        vf,
        Shape {
            batch: 1,
            channels: kv_heads,
            height: hd,
            width: seq,
        },
    );
    let v_base = g.transpose(v4, [0, 1, 3, 2]);
    let v = tile_kv_heads(&mut g, v_base, kv_heads, gqa_ratio, seq, hd);

    // ── RoPE ──
    let (cos_data, sin_data) = rope_table(seq, hd);
    let rope_cos = g.constant(
        &cos_data,
        Shape {
            batch: 1,
            channels: 1,
            height: seq,
            width: hd,
        },
    );
    let rope_sin = g.constant(
        &sin_data,
        Shape {
            batch: 1,
            channels: 1,
            height: seq,
            width: hd,
        },
    );

    // rotate_half(q): split pairs, negate odd, concat reversed, reshape back
    let pairs_q = seq * hd / 2;
    let q_p = g.reshape(
        q,
        Shape {
            batch: 1,
            channels: heads,
            height: pairs_q,
            width: 2,
        },
    );
    let q_e = g.slice(q_p, [0, 0, 0, 0], [1, heads, pairs_q, 1]); // even
    let q_o = g.slice(q_p, [0, 0, 0, 1], [1, heads, pairs_q, 1]); // odd
    let neg1 = g.constant_with_scalar(
        -1.0,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let nq = g.multiplication(q_o, neg1);
    let qrp = g.concat(&[nq, q_e], 3); // [-odd, even] → rotated
    let q_rot = g.reshape(
        qrp,
        Shape {
            batch: 1,
            channels: heads,
            height: seq,
            width: hd,
        },
    );
    let qc = g.multiplication(q, rope_cos);
    let qrs = g.multiplication(q_rot, rope_sin);
    let q_rope = g.addition(qc, qrs);

    // rotate_half(k): same pattern
    let pairs_k = seq * hd / 2;
    let k_p = g.reshape(
        k,
        Shape {
            batch: 1,
            channels: heads,
            height: pairs_k,
            width: 2,
        },
    );
    let k_e = g.slice(k_p, [0, 0, 0, 0], [1, heads, pairs_k, 1]);
    let k_o = g.slice(k_p, [0, 0, 0, 1], [1, heads, pairs_k, 1]);
    let nk = g.multiplication(k_o, neg1);
    let krp = g.concat(&[nk, k_e], 3);
    let k_rot = g.reshape(
        krp,
        Shape {
            batch: 1,
            channels: heads,
            height: seq,
            width: hd,
        },
    );
    let kc = g.multiplication(k, rope_cos);
    let krs = g.multiplication(k_rot, rope_sin);
    let k_rope = g.addition(kc, krs);

    // ── Attention: Q_rope @ K_rope^T * scale + mask → softmax → @ V ──
    // scores = Q_rope @ K_rope^T: [1,HEADS,SEQ,HD] @ [1,HEADS,HD,SEQ] → [1,HEADS,SEQ,SEQ]
    let sc1 = g.matrix_multiplication(q_rope, k_rope, false, true);
    let scv = g.constant_with_scalar(
        scale,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let sc2 = g.multiplication(sc1, scv);

    // Causal mask
    let mask_data = causal_mask(seq);
    let cm = g.constant(
        &mask_data,
        Shape {
            batch: 1,
            channels: 1,
            height: seq,
            width: seq,
        },
    );
    let ms = g.addition(sc2, cm);

    // Softmax along last axis
    let aw = g.soft_max(ms, -1);

    // Attention output: scores @ V → [1,HEADS,SEQ,HD]
    let a4 = g.matrix_multiplication(aw, v, false, false);

    // Reshape attn_out to [1,Q_DIM,1,SEQ]
    let at = g.transpose(a4, [0, 1, 3, 2]); // [1,HEADS,HD,SEQ]
    let af = g.reshape(
        at,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );

    // ── Flatten Q_rope, K_rope back to [1,X,1,SEQ] for backward pass ──
    let qrt = g.transpose(q_rope, [0, 1, 3, 2]); // [1,HEADS,HD,SEQ]
    let qrf = g.reshape(
        qrt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );
    let krt = g.transpose(k_rope, [0, 1, 3, 2]);
    let krf = g.reshape(
        krt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );
    let vt = g.transpose(v, [0, 1, 3, 2]);
    let vf_tiled = g.reshape(
        vt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );

    // ── Output: concat(attn_out, Q_rope, K_rope, V, xnorm_passthrough) ──
    let _out = g.concat(&[af, qrf, krf, vf_tiled, xn], 1);

    g
}

/// Build the SDPA forward graph from a packed input tensor.
pub fn build(cfg: &ModelConfig) -> Graph {
    build_impl(cfg, false)
}

/// Build the SDPA forward graph from split runtime inputs:
/// xnorm, Wq, Wk, Wv.
pub fn build_split(cfg: &ModelConfig) -> Graph {
    build_impl(cfg, true)
}

/// Input spatial width for sdpaFwd.
pub fn input_spatial_width(cfg: &ModelConfig) -> usize {
    cfg.seq + cfg.q_dim + cfg.kv_dim + cfg.kv_dim
}

/// Output channel count for sdpaFwd.
pub fn output_channels(cfg: &ModelConfig) -> usize {
    cfg.q_dim + cfg.q_dim + cfg.q_dim + cfg.q_dim + cfg.dim
}
