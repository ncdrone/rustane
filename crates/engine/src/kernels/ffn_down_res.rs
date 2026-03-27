//! FFN down projection kernel (Part B of split ffnFused).
//!
//! Separated from ffnFused to keep ANE tensor dimensions under ~16,384.
//! Used when 2*seq + 3*hidden > 16384 (i.e., 1B+ models).
//!
//! Input IOSurface: [1, HIDDEN, 1, SEQ + DIM]
//!   sp[0:SEQ]           = gate_out [HIDDEN, SEQ]   (from ffn_gate_up kernel)
//!   sp[SEQ:SEQ+DIM]     = W2^T [HIDDEN, DIM]       (pre-transposed on CPU before staging)
//!
//! Output: [1, DIM, 1, SEQ]
//!   = ffn_out = gate_out @ W2^T
//!
//! NOTE: Residual addition (x_next = x2 + alpha * ffn_out) is done on CPU after this kernel.
//! This keeps the kernel simple and avoids mixed-channel-count packing.

use ane_bridge::ane::{Graph, Shape};
use crate::model::ModelConfig;

/// Build the down projection graph.
pub fn build(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let hidden = cfg.hidden;

    let sp_in = seq + dim;

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: hidden, height: 1, width: sp_in });

    // ── Slice inputs ──
    let gate_out = g.slice(input, [0, 0, 0, 0], [1, hidden, 1, seq]);
    let w2t = g.slice(input, [0, 0, 0, seq], [1, hidden, 1, dim]);

    // ── Down projection: gate_out @ W2^T → [DIM, SEQ] ──
    // gate_out: [HIDDEN, SEQ], W2^T: [HIDDEN, DIM]
    let gm = g.reshape(gate_out, Shape { batch: 1, channels: 1, height: hidden, width: seq });
    let gmt = g.transpose(gm, [0, 1, 3, 2]); // [1,1,SEQ,HIDDEN]
    let w2m = g.reshape(w2t, Shape { batch: 1, channels: 1, height: hidden, width: dim });
    // [1,1,SEQ,HIDDEN] @ [1,1,HIDDEN,DIM] → [1,1,SEQ,DIM]
    let fm = g.matrix_multiplication(gmt, w2m, false, false);
    let ft = g.transpose(fm, [0, 1, 3, 2]);
    let ffn_out = g.reshape(ft, Shape { batch: 1, channels: dim, height: 1, width: seq });

    let _out = ffn_out;

    g
}

/// Input spatial width for ffn_down_res.
pub fn input_spatial_width(cfg: &ModelConfig) -> usize {
    cfg.seq + cfg.dim
}

/// Output channel count for ffn_down_res.
pub fn output_channels(cfg: &ModelConfig) -> usize {
    cfg.dim
}
