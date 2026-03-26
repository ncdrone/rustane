//! SDPA Forward kernel: QKV projection + RoPE + attention + output.
//!
//! RoPE uses matmul-based permutation (M3/M4 Ultra compatible) instead of
//! width-axis reshape/slice/concat (M4 Max only).
//!
//! Inputs:
//! - xnorm: [1, DIM, 1, SEQ]
//! - wq:    [1, DIM, 1, Q_DIM]
//! - wk:    [1, DIM, 1, KV_DIM]
//! - wv:    [1, DIM, 1, KV_DIM]
//!
//! Outputs:
//! - attn_out: [1, Q_DIM, 1, SEQ]
//! - q_rope:   [1, Q_DIM, 1, SEQ]
//! - k_rope:   [1, KV_DIM, 1, SEQ]
//! - v:        [1, KV_DIM, 1, SEQ]
//!
//! For gpt_karpathy (MHA): Q_DIM = KV_DIM = DIM = 768, HEADS = KV_HEADS = 6

use ane_bridge::ane::{Graph, Shape};

use crate::model::ModelConfig;

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

fn rope_permutation_matrix(hd: usize) -> Vec<f32> {
    let mut perm = vec![0.0f32; hd * hd];
    for i in 0..hd / 2 {
        perm[(2 * i + 1) * hd + 2 * i] = -1.0;
        perm[(2 * i) * hd + 2 * i + 1] = 1.0;
    }
    perm
}

pub fn build(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let heads = cfg.heads;
    let hd = cfg.hd;

    let scale = 1.0 / (hd as f32).sqrt();

    let mut g = Graph::new();
    let xnorm = g.placeholder(xnorm_shape(cfg));
    let wq = g.placeholder(wq_shape(cfg));
    let wk = g.placeholder(wk_shape(cfg));
    let wv = g.placeholder(wv_shape(cfg));

    let xn2 = g.reshape(
        xnorm,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: seq,
        },
    );
    let xnt = g.transpose(xn2, [0, 1, 3, 2]);

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

    let qm = g.matrix_multiplication(xnt, wq2, false, false);
    let km = g.matrix_multiplication(xnt, wk2, false, false);
    let vm = g.matrix_multiplication(xnt, wv2, false, false);

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

    let q4 = g.reshape(
        qf,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let q = g.transpose(q4, [0, 1, 3, 2]);
    let k4 = g.reshape(
        kf,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let k = g.transpose(k4, [0, 1, 3, 2]);
    let v4 = g.reshape(
        vf,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let v = g.transpose(v4, [0, 1, 3, 2]);

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
    let perm_data = rope_permutation_matrix(hd);
    let perm = g.constant(
        &perm_data,
        Shape {
            batch: 1,
            channels: 1,
            height: hd,
            width: hd,
        },
    );

    let q_rot = g.matrix_multiplication(q, perm, false, false);
    let qc = g.multiplication(q, rope_cos);
    let qrs = g.multiplication(q_rot, rope_sin);
    let q_rope = g.addition(qc, qrs);

    let k_rot = g.matrix_multiplication(k, perm, false, false);
    let kc = g.multiplication(k, rope_cos);
    let krs = g.multiplication(k_rot, rope_sin);
    let k_rope = g.addition(kc, krs);

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
    let aw = g.soft_max(ms, -1);
    let a4 = g.matrix_multiplication(aw, v, false, false);

    let at = g.transpose(a4, [0, 1, 3, 2]);
    let af = g.reshape(
        at,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );

    let qrt = g.transpose(q_rope, [0, 1, 3, 2]);
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
            channels: kv_dim,
            height: 1,
            width: seq,
        },
    );

    let _attn_out = g.slice(af, [0, 0, 0, 0], [1, q_dim, 1, seq]);
    let _q_rope = g.slice(qrf, [0, 0, 0, 0], [1, q_dim, 1, seq]);
    let _k_rope = g.slice(krf, [0, 0, 0, 0], [1, kv_dim, 1, seq]);
    let _v = g.slice(vf, [0, 0, 0, 0], [1, kv_dim, 1, seq]);

    g
}

pub fn xnorm_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.seq,
    }
}

pub fn wq_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.q_dim,
    }
}

pub fn wk_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.kv_dim,
    }
}

pub fn wv_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.dim,
        height: 1,
        width: cfg.kv_dim,
    }
}

pub fn attn_out_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.q_dim,
        height: 1,
        width: cfg.seq,
    }
}

pub fn q_rope_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.q_dim,
        height: 1,
        width: cfg.seq,
    }
}

pub fn k_rope_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.kv_dim,
        height: 1,
        width: cfg.seq,
    }
}

pub fn v_shape(cfg: &ModelConfig) -> Shape {
    Shape {
        batch: 1,
        channels: cfg.kv_dim,
        height: 1,
        width: cfg.seq,
    }
}
