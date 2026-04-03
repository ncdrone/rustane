//! Fused FFN forward kernel: configurable gate activation + up + down + residual.
//!
//! Input IOSurface: [1, DIM, 1, 2*SEQ + 3*HIDDEN]
//!   sp[0:SEQ]                     = x2norm [DIM, SEQ]  (post-RMSNorm input)
//!   sp[SEQ:2*SEQ]                 = x2 [DIM, SEQ]      (pre-norm residual)
//!   sp[2*SEQ:2*SEQ+HIDDEN]        = W1 [DIM, HIDDEN]   (gate projection)
//!   sp[2*SEQ+HIDDEN:2*SEQ+2*HID]  = W3 [DIM, HIDDEN]   (up projection)
//!   sp[2*SEQ+2*HID:...]           = W2 [DIM, HIDDEN]   (down projection, stored as [DIM,HIDDEN])
//!
//! Output: [1, DIM + 3*HIDDEN, 1, SEQ]
//!   = concat(x_next, h1, h3, gate_out)
//!   where x_next = x2 + alpha * (gate_out @ W2^T)
//!
//! h1 = xnorm @ W1 (gate), h3 = xnorm @ W3 (up)
//! gate_out = activation(h1) * h3
//! ffn_out = gate_out @ W2^T  (W2 stored as [DIM,HIDDEN], transposed inside kernel)

use crate::model::{FfnActivation, ModelConfig};
use ane_bridge::ane::{Graph, Shape};

fn build_impl(cfg: &ModelConfig, split_inputs: bool) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let hidden = cfg.hidden;
    let nlayers = cfg.nlayers;

    let sp_in = 2 * seq + 3 * hidden;
    let alpha = 1.0 / (2.0 * nlayers as f32).sqrt(); // DeepNet residual scaling

    let mut g = Graph::new();
    let input = if split_inputs {
        let x2norm = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let x2 = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        });
        let w1 = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: hidden,
        });
        let w3 = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: hidden,
        });
        let w2 = g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: hidden,
        });
        let lhs = g.concat(&[x2norm, x2], 3);
        let rhs = g.concat(&[w1, w3, w2], 3);
        g.concat(&[lhs, rhs], 3)
    } else {
        g.placeholder(Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: sp_in,
        })
    };

    // ── Slice inputs ──
    let x2norm = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
    let x2 = g.slice(input, [0, 0, 0, seq], [1, dim, 1, seq]);
    let w1 = g.slice(input, [0, 0, 0, 2 * seq], [1, dim, 1, hidden]);
    let w3 = g.slice(input, [0, 0, 0, 2 * seq + hidden], [1, dim, 1, hidden]);
    let w2_raw = g.slice(input, [0, 0, 0, 2 * seq + 2 * hidden], [1, dim, 1, hidden]);

    // ── Gate and up projections: xnorm @ W1, xnorm @ W3 ──
    // Reshape for matmul: [1,DIM,1,SEQ] → [1,1,DIM,SEQ] → [1,1,SEQ,DIM]
    let xn2 = g.reshape(
        x2norm,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: seq,
        },
    );
    let xnt = g.transpose(xn2, [0, 1, 3, 2]);

    let w12 = g.reshape(
        w1,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: hidden,
        },
    );
    let w32 = g.reshape(
        w3,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: hidden,
        },
    );

    // [1,1,SEQ,DIM] @ [1,1,DIM,HIDDEN] → [1,1,SEQ,HIDDEN]
    let h1m = g.matrix_multiplication(xnt, w12, false, false);
    let h3m = g.matrix_multiplication(xnt, w32, false, false);

    // Reshape back to [1,HIDDEN,1,SEQ]
    let h1t = g.transpose(h1m, [0, 1, 3, 2]);
    let h3t = g.transpose(h3m, [0, 1, 3, 2]);
    let h1 = g.reshape(
        h1t,
        Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: seq,
        },
    );
    let h3 = g.reshape(
        h3t,
        Shape {
            batch: 1,
            channels: hidden,
            height: 1,
            width: seq,
        },
    );

    // ── Gate activation: configurable activation(h1) * h3 ──
    let gate_act = match cfg.ffn_activation {
        FfnActivation::SwiGlu => {
            let sig = g.sigmoid(h1);
            g.multiplication(h1, sig)
        }
        FfnActivation::LeakyReluSq => {
            let leaky = g.leaky_relu(h1, 0.5);
            g.multiplication(leaky, leaky)
        }
    };
    let gate = g.multiplication(gate_act, h3);

    // ── Down projection: gate @ W2^T ──
    // W2 stored as [DIM, HIDDEN], need transpose for gate[HIDDEN,SEQ] @ W2^T → [DIM,SEQ]
    let g2 = g.reshape(
        gate,
        Shape {
            batch: 1,
            channels: 1,
            height: hidden,
            width: seq,
        },
    );
    let gt = g.transpose(g2, [0, 1, 3, 2]); // [1,1,SEQ,HIDDEN]
    let w22 = g.reshape(
        w2_raw,
        Shape {
            batch: 1,
            channels: 1,
            height: dim,
            width: hidden,
        },
    );
    let w2t = g.transpose(w22, [0, 1, 3, 2]); // [1,1,HIDDEN,DIM]
    let fm = g.matrix_multiplication(gt, w2t, false, false); // [1,1,SEQ,DIM]
    let ft = g.transpose(fm, [0, 1, 3, 2]);
    let ffn_out = g.reshape(
        ft,
        Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: seq,
        },
    );

    // ── Residual: x_next = x2 + alpha * ffn_out ──
    let alpha_const = g.constant_with_scalar(
        alpha,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let ffn_scaled = g.multiplication(ffn_out, alpha_const);
    let x_next = g.addition(x2, ffn_scaled);

    // ── Output: concat(x_next, h1, h3, gate) for backward pass caching ──
    let _out = g.concat(&[x_next, h1, h3, gate], 1);

    g
}

/// Build the fused FFN forward graph from a packed input tensor.
pub fn build(cfg: &ModelConfig) -> Graph {
    build_impl(cfg, false)
}

/// Build the fused FFN forward graph from split runtime inputs:
/// x2norm, x2, W1, W3, W2.
pub fn build_split(cfg: &ModelConfig) -> Graph {
    build_impl(cfg, true)
}

/// Input spatial width for ffnFused.
pub fn input_spatial_width(cfg: &ModelConfig) -> usize {
    2 * cfg.seq + 3 * cfg.hidden
}

/// Output channel count for ffnFused.
pub fn output_channels(cfg: &ModelConfig) -> usize {
    cfg.dim + 3 * cfg.hidden
}
