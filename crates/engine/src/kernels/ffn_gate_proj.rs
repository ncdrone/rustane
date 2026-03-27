//! FFN gate projection kernel (W1 only).
//!
//! Part of the 3-way FFN split for 5B+ models where even the 2-way
//! gate+up kernel exceeds ANE's ~16,384 dimension limit.
//!
//! Input IOSurface: [1, DIM, 1, SEQ + HIDDEN]
//!   sp[0:SEQ]           = x2norm [DIM, SEQ]   (post-RMSNorm input)
//!   sp[SEQ:SEQ+HIDDEN]  = W1 [DIM, HIDDEN]    (gate projection weights)
//!
//! Output: [1, HIDDEN, 1, SEQ]
//!   = h1 = x2norm @ W1

use ane_bridge::ane::{Graph, Shape};
use crate::model::ModelConfig;

/// Build the gate projection graph.
pub fn build(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let hidden = cfg.hidden;

    let sp_in = seq + hidden;

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp_in });

    // ── Slice inputs ──
    let x2norm = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
    let w1 = g.slice(input, [0, 0, 0, seq], [1, dim, 1, hidden]);

    // ── Gate projection: xnorm @ W1 → h1 ──
    let xn2 = g.reshape(x2norm, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let xnt = g.transpose(xn2, [0, 1, 3, 2]); // [1,1,SEQ,DIM]
    let w12 = g.reshape(w1, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    // [1,1,SEQ,DIM] @ [1,1,DIM,HIDDEN] → [1,1,SEQ,HIDDEN]
    let h1m = g.matrix_multiplication(xnt, w12, false, false);
    let h1t = g.transpose(h1m, [0, 1, 3, 2]);
    let h1 = g.reshape(h1t, Shape { batch: 1, channels: hidden, height: 1, width: seq });

    let _out = h1;

    g
}

/// Input spatial width for ffn_gate_proj.
pub fn input_spatial_width(cfg: &ModelConfig) -> usize {
    cfg.seq + cfg.hidden
}

/// Output channel count for ffn_gate_proj.
pub fn output_channels(_cfg: &ModelConfig) -> usize {
    _cfg.hidden
}
