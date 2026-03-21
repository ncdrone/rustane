//! FFN up projection kernel (W3 only).
//!
//! Part of the 3-way FFN split for 5B+ models where even the 2-way
//! gate+up kernel exceeds ANE's ~16,384 dimension limit.
//!
//! Input IOSurface: [1, DIM, 1, SEQ + HIDDEN]
//!   sp[0:SEQ]           = x2norm [DIM, SEQ]   (post-RMSNorm input)
//!   sp[SEQ:SEQ+HIDDEN]  = W3 [DIM, HIDDEN]    (up projection weights)
//!
//! Output: [1, HIDDEN, 1, SEQ]
//!   = h3 = x2norm @ W3

use ane_bridge::ane::{Graph, Shape};
use crate::model::ModelConfig;

/// Build the up projection graph.
pub fn build(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let hidden = cfg.hidden;

    let sp_in = seq + hidden;

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp_in });

    // ── Slice inputs ──
    let x2norm = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
    let w3 = g.slice(input, [0, 0, 0, seq], [1, dim, 1, hidden]);

    // ── Up projection: xnorm @ W3 → h3 ──
    let xn2 = g.reshape(x2norm, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let xnt = g.transpose(xn2, [0, 1, 3, 2]); // [1,1,SEQ,DIM]
    let w32 = g.reshape(w3, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    // [1,1,SEQ,DIM] @ [1,1,DIM,HIDDEN] → [1,1,SEQ,HIDDEN]
    let h3m = g.matrix_multiplication(xnt, w32, false, false);
    let h3t = g.transpose(h3m, [0, 1, 3, 2]);
    let h3 = g.reshape(h3t, Shape { batch: 1, channels: hidden, height: 1, width: seq });

    let _out = h3;

    g
}

/// Input spatial width for ffn_up_proj.
pub fn input_spatial_width(cfg: &ModelConfig) -> usize {
    cfg.seq + cfg.hidden
}

/// Output channel count for ffn_up_proj.
pub fn output_channels(_cfg: &ModelConfig) -> usize {
    _cfg.hidden
}
