//! FFN gate+up kernel (Part A of split ffnFused).
//!
//! Separated from ffnFused to keep ANE tensor dimensions under ~16,384.
//! Used when dim + 3*hidden > 16384 (i.e., 1B+ models).
//!
//! Input IOSurface: [1, DIM, 1, SEQ + 2*HIDDEN]
//!   sp[0:SEQ]                     = x2norm [DIM, SEQ]   (post-RMSNorm input)
//!   sp[SEQ:SEQ+HIDDEN]            = W1 [DIM, HIDDEN]    (gate projection)
//!   sp[SEQ+HIDDEN:SEQ+2*HIDDEN]   = W3 [DIM, HIDDEN]    (up projection)
//!
//! Output: [1, HIDDEN, 1, SEQ]
//!   = gate_out = silu(xnorm @ W1) * (xnorm @ W3)

use ane_bridge::ane::{Graph, Shape};
use crate::model::ModelConfig;

/// Build the gate+up projection graph.
pub fn build(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let hidden = cfg.hidden;

    let sp_in = seq + 2 * hidden;

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp_in });

    // ── Slice inputs ──
    let x2norm = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
    let w1 = g.slice(input, [0, 0, 0, seq], [1, dim, 1, hidden]);
    let w3 = g.slice(input, [0, 0, 0, seq + hidden], [1, dim, 1, hidden]);

    // ── Gate and up projections: xnorm @ W1, xnorm @ W3 ──
    let xn2 = g.reshape(x2norm, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let xnt = g.transpose(xn2, [0, 1, 3, 2]);

    let w12 = g.reshape(w1, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let w32 = g.reshape(w3, Shape { batch: 1, channels: 1, height: dim, width: hidden });

    // [1,1,SEQ,DIM] @ [1,1,DIM,HIDDEN] → [1,1,SEQ,HIDDEN]
    let h1m = g.matrix_multiplication(xnt, w12, false, false);
    let h3m = g.matrix_multiplication(xnt, w32, false, false);

    // Reshape back to [1,HIDDEN,1,SEQ]
    let h1t = g.transpose(h1m, [0, 1, 3, 2]);
    let h3t = g.transpose(h3m, [0, 1, 3, 2]);
    let h1 = g.reshape(h1t, Shape { batch: 1, channels: hidden, height: 1, width: seq });
    let h3 = g.reshape(h3t, Shape { batch: 1, channels: hidden, height: 1, width: seq });

    // ── SiLU gate: silu(h1) * h3 ──
    let sig = g.sigmoid(h1);
    let silu = g.multiplication(h1, sig);
    let gate = g.multiplication(silu, h3);

    // Output: gate_out [HIDDEN, SEQ]
    let _out = gate;

    g
}

/// Input spatial width for ffn_gate_up.
pub fn input_spatial_width(cfg: &ModelConfig) -> usize {
    cfg.seq + 2 * cfg.hidden
}

/// Output channel count for ffn_gate_up.
pub fn output_channels(cfg: &ModelConfig) -> usize {
    cfg.hidden
}
