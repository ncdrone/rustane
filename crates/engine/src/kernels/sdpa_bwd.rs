//! SDPA backward kernels: sdpaBwd1 (dV, dScores) and sdpaBwd2 (dQ, dK).
//!
//! Both are weight-free — activations packed in channels, not spatial.
//!
//! sdpaBwd1 Input: [1, 4*Q_DIM, 1, SEQ]
//!   ch[0:Q_DIM]       = Q_rope
//!   ch[Q_DIM:2*Q_DIM] = K_tiled (= K_rope for MHA)
//!   ch[2*Q_DIM:3*Q_DIM] = V_tiled (= V for MHA)
//!   ch[3*Q_DIM:4*Q_DIM] = da (from wotBwd)
//!
//! sdpaBwd1 Output: [1, Q_DIM + 2*SCORE_CH, 1, SEQ]
//!   ch[0:Q_DIM]                     = dV
//!   ch[Q_DIM:Q_DIM+SCORE_CH]       = probs (for sdpaBwd2)
//!   ch[Q_DIM+SCORE_CH:Q_DIM+2*SC]  = dp (for sdpaBwd2)
//!   where SCORE_CH = HEADS * SEQ
//!
//! sdpaBwd2 Input: [1, 2*SCORE_CH + 2*Q_DIM, 1, SEQ]
//!   ch[0:SCORE_CH]                  = probs
//!   ch[SCORE_CH:2*SCORE_CH]         = dp
//!   ch[2*SCORE_CH:2*SCORE_CH+Q_DIM] = Q_rope
//!   ch[2*SCORE_CH+Q_DIM:...]        = K_tiled
//!
//! sdpaBwd2 Output: [1, 2*Q_DIM, 1, SEQ]
//!   ch[0:Q_DIM]       = dQ
//!   ch[Q_DIM:2*Q_DIM] = dK

use crate::model::ModelConfig;
use ane_bridge::ane::{Graph, Shape};

/// Generate causal mask (same as sdpa_fwd).
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

/// Build sdpaBwd1: recompute attention, compute dV and dp.
pub fn build_bwd1(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let score_ch = heads * seq;

    let in_ch = 4 * q_dim;
    let mut g = Graph::new();
    let input = g.placeholder(Shape {
        batch: 1,
        channels: in_ch,
        height: 1,
        width: seq,
    });

    // Slice Q_rope, K_tiled, V_tiled, da from channels
    let q_flat = g.slice(input, [0, 0, 0, 0], [1, q_dim, 1, seq]);
    let k_flat = g.slice(input, [0, q_dim, 0, 0], [1, q_dim, 1, seq]);
    let v_flat = g.slice(input, [0, 2 * q_dim, 0, 0], [1, q_dim, 1, seq]);
    let da_flat = g.slice(input, [0, 3 * q_dim, 0, 0], [1, q_dim, 1, seq]);

    // Reshape to [1, HEADS, HD, SEQ] then transpose to [1, HEADS, SEQ, HD]
    let q4 = g.reshape(
        q_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let k4 = g.reshape(
        k_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let v4 = g.reshape(
        v_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let da4 = g.reshape(
        da_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );

    let q = g.transpose(q4, [0, 1, 3, 2]); // [1, HEADS, SEQ, HD]
    let k = g.transpose(k4, [0, 1, 3, 2]);
    let v = g.transpose(v4, [0, 1, 3, 2]);
    let da = g.transpose(da4, [0, 1, 3, 2]);

    // Recompute attention: Q @ K^T * scale + mask → softmax
    let sc1 = g.matrix_multiplication(q, k, false, true); // [1, HEADS, SEQ, SEQ]
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
    let probs = g.soft_max(ms, -1); // [1, HEADS, SEQ, SEQ]

    // dV = probs^T @ da: [1,H,SEQ,SEQ]^T @ [1,H,SEQ,HD] → [1,H,SEQ,HD]
    let dv4 = g.matrix_multiplication(probs, da, true, false);

    // dp = da @ V^T: [1,H,SEQ,HD] @ [1,H,HD,SEQ] → [1,H,SEQ,SEQ]
    let dp = g.matrix_multiplication(da, v, false, true);

    // Flatten dV to [1, Q_DIM, 1, SEQ]
    let dvt = g.transpose(dv4, [0, 1, 3, 2]); // [1, HEADS, HD, SEQ]
    let dv_flat = g.reshape(
        dvt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );

    // Flatten probs: [1, HEADS, SEQ, SEQ] → [1, SCORE_CH, 1, SEQ]
    let probs_flat = g.reshape(
        probs,
        Shape {
            batch: 1,
            channels: score_ch,
            height: 1,
            width: seq,
        },
    );

    // Flatten dp: [1, HEADS, SEQ, SEQ] → [1, SCORE_CH, 1, SEQ]
    let dp_flat = g.reshape(
        dp,
        Shape {
            batch: 1,
            channels: score_ch,
            height: 1,
            width: seq,
        },
    );

    // Output: concat(dV, probs, dp) along channels
    let _out = g.concat(&[dv_flat, probs_flat, dp_flat], 1);

    g
}

/// Build sdpaBwd1 with four split runtime inputs.
///
/// Inputs:
///   q_flat:  [1, Q_DIM, 1, SEQ] = Q_rope
///   k_flat:  [1, Q_DIM, 1, SEQ] = K_tiled
///   v_flat:  [1, Q_DIM, 1, SEQ] = V_tiled
///   da_flat: [1, Q_DIM, 1, SEQ] = da
///
/// Output: [1, Q_DIM + 2*SCORE_CH, 1, SEQ] = concat(dV, probs, dp)
pub fn build_bwd1_split(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let score_ch = heads * seq;

    let mut g = Graph::new();
    let q_flat = g.placeholder(Shape {
        batch: 1,
        channels: q_dim,
        height: 1,
        width: seq,
    });
    let k_flat = g.placeholder(Shape {
        batch: 1,
        channels: q_dim,
        height: 1,
        width: seq,
    });
    let v_flat = g.placeholder(Shape {
        batch: 1,
        channels: q_dim,
        height: 1,
        width: seq,
    });
    let da_flat = g.placeholder(Shape {
        batch: 1,
        channels: q_dim,
        height: 1,
        width: seq,
    });

    let q4 = g.reshape(
        q_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let k4 = g.reshape(
        k_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let v4 = g.reshape(
        v_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let da4 = g.reshape(
        da_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );

    let q = g.transpose(q4, [0, 1, 3, 2]);
    let k = g.transpose(k4, [0, 1, 3, 2]);
    let v = g.transpose(v4, [0, 1, 3, 2]);
    let da = g.transpose(da4, [0, 1, 3, 2]);

    let sc1 = g.matrix_multiplication(q, k, false, true);
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
    let probs = g.soft_max(ms, -1);

    let dv4 = g.matrix_multiplication(probs, da, true, false);
    let dp = g.matrix_multiplication(da, v, false, true);

    let dvt = g.transpose(dv4, [0, 1, 3, 2]);
    let dv_flat = g.reshape(
        dvt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );
    let probs_flat = g.reshape(
        probs,
        Shape {
            batch: 1,
            channels: score_ch,
            height: 1,
            width: seq,
        },
    );
    let dp_flat = g.reshape(
        dp,
        Shape {
            batch: 1,
            channels: score_ch,
            height: 1,
            width: seq,
        },
    );

    let _out = g.concat(&[dv_flat, probs_flat, dp_flat], 1);

    g
}

/// Build sdpaBwd2: softmax gradient, compute dQ and dK.
pub fn build_bwd2(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let score_ch = heads * seq;

    let in_ch = 2 * score_ch + 2 * q_dim;
    let mut g = Graph::new();
    let input = g.placeholder(Shape {
        batch: 1,
        channels: in_ch,
        height: 1,
        width: seq,
    });

    // Slice probs, dp, Q_rope, K_tiled from channels
    let probs_flat = g.slice(input, [0, 0, 0, 0], [1, score_ch, 1, seq]);
    let dp_flat = g.slice(input, [0, score_ch, 0, 0], [1, score_ch, 1, seq]);
    let q_flat = g.slice(input, [0, 2 * score_ch, 0, 0], [1, q_dim, 1, seq]);
    let k_flat = g.slice(input, [0, 2 * score_ch + q_dim, 0, 0], [1, q_dim, 1, seq]);

    // Unflatten probs and dp: [1, SCORE_CH, 1, SEQ] → [1, HEADS, SEQ, SEQ]
    let probs = g.reshape(
        probs_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: seq,
            width: seq,
        },
    );
    let dp = g.reshape(
        dp_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: seq,
            width: seq,
        },
    );

    // Reshape Q, K to heads: [1, Q_DIM, 1, SEQ] → [1, HEADS, HD, SEQ] → [1, HEADS, SEQ, HD]
    let q4 = g.reshape(
        q_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let q = g.transpose(q4, [0, 1, 3, 2]);
    let k4 = g.reshape(
        k_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let k = g.transpose(k4, [0, 1, 3, 2]);

    // Softmax gradient:
    // pdp = probs * dp
    let pdp = g.multiplication(probs, dp);
    // spdp = reduce_sum(pdp, axis=-1)  → [1, HEADS, SEQ, 1]
    let spdp = g.reduce_sum(pdp, -1);
    // dps = dp - spdp  (broadcasts over last dim)
    let dps = g.subtraction(dp, spdp);
    // ds0 = probs * dps
    let ds0 = g.multiplication(probs, dps);
    // ds = ds0 * scale
    let scv = g.constant_with_scalar(
        scale,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let ds = g.multiplication(ds0, scv);

    // dQ = ds @ K: [1,H,SEQ,SEQ] @ [1,H,SEQ,HD] → [1,H,SEQ,HD]
    let dq4 = g.matrix_multiplication(ds, k, false, false);
    // dK = ds^T @ Q: [1,H,SEQ,SEQ]^T @ [1,H,SEQ,HD] → [1,H,SEQ,HD]
    let dk4 = g.matrix_multiplication(ds, q, true, false);

    // Flatten to [1, Q_DIM, 1, SEQ]
    let dqt = g.transpose(dq4, [0, 1, 3, 2]); // [1, HEADS, HD, SEQ]
    let dq_flat = g.reshape(
        dqt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );
    let dkt = g.transpose(dk4, [0, 1, 3, 2]);
    let dk_flat = g.reshape(
        dkt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );

    // Output: concat(dQ, dK)
    let _out = g.concat(&[dq_flat, dk_flat], 1);

    g
}

/// Build sdpaBwd2 reusing sdpaBwd1's output IOSurface directly.
///
/// Inputs:
///   bwd1_out: [1, Q_DIM + 2*SCORE_CH, 1, SEQ]
///     ch[0:Q_DIM]                    = dV     (ignored here)
///     ch[Q_DIM:Q_DIM+SCORE_CH]       = probs
///     ch[Q_DIM+SCORE_CH:...]         = dp
///   q_flat:   [1, Q_DIM, 1, SEQ]     = Q_rope
///   k_flat:   [1, Q_DIM, 1, SEQ]     = K_tiled
///
/// Output: [1, 2*Q_DIM, 1, SEQ] = concat(dQ, dK)
pub fn build_bwd2_split_from_bwd1(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let score_ch = heads * seq;

    let mut g = Graph::new();
    let bwd1_out = g.placeholder(Shape {
        batch: 1,
        channels: q_dim + 2 * score_ch,
        height: 1,
        width: seq,
    });
    let q_flat = g.placeholder(Shape {
        batch: 1,
        channels: q_dim,
        height: 1,
        width: seq,
    });
    let k_flat = g.placeholder(Shape {
        batch: 1,
        channels: q_dim,
        height: 1,
        width: seq,
    });

    // Slice probs and dp directly from sdpaBwd1 output.
    let probs_flat = g.slice(bwd1_out, [0, q_dim, 0, 0], [1, score_ch, 1, seq]);
    let dp_flat = g.slice(bwd1_out, [0, q_dim + score_ch, 0, 0], [1, score_ch, 1, seq]);

    let probs = g.reshape(
        probs_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: seq,
            width: seq,
        },
    );
    let dp = g.reshape(
        dp_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: seq,
            width: seq,
        },
    );

    let q4 = g.reshape(
        q_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let q = g.transpose(q4, [0, 1, 3, 2]);
    let k4 = g.reshape(
        k_flat,
        Shape {
            batch: 1,
            channels: heads,
            height: hd,
            width: seq,
        },
    );
    let k = g.transpose(k4, [0, 1, 3, 2]);

    let pdp = g.multiplication(probs, dp);
    let spdp = g.reduce_sum(pdp, -1);
    let dps = g.subtraction(dp, spdp);
    let ds0 = g.multiplication(probs, dps);
    let scv = g.constant_with_scalar(
        scale,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let ds = g.multiplication(ds0, scv);

    let dq4 = g.matrix_multiplication(ds, k, false, false);
    let dk4 = g.matrix_multiplication(ds, q, true, false);

    let dqt = g.transpose(dq4, [0, 1, 3, 2]);
    let dq_flat = g.reshape(
        dqt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );
    let dkt = g.transpose(dk4, [0, 1, 3, 2]);
    let dk_flat = g.reshape(
        dkt,
        Shape {
            batch: 1,
            channels: q_dim,
            height: 1,
            width: seq,
        },
    );

    let _out = g.concat(&[dq_flat, dk_flat], 1);

    g
}

/// Input channel count for sdpaBwd1.
pub fn bwd1_input_channels(cfg: &ModelConfig) -> usize {
    4 * cfg.q_dim
}

/// Output channel count for sdpaBwd1.
pub fn bwd1_output_channels(cfg: &ModelConfig) -> usize {
    cfg.q_dim + 2 * cfg.heads * cfg.seq
}

/// Input channel count for sdpaBwd2.
pub fn bwd2_input_channels(cfg: &ModelConfig) -> usize {
    2 * cfg.heads * cfg.seq + 2 * cfg.q_dim
}

/// Output channel count for sdpaBwd2.
pub fn bwd2_output_channels(cfg: &ModelConfig) -> usize {
    2 * cfg.q_dim
}
