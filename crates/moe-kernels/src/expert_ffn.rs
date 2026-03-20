//! Expert FFN: SwiGLU feed-forward network for MoE.
//!
//! Each expert has 3 weight matrices:
//! - W1 (gate): [dim, hidden] — gate projection
//! - W3 (up):   [dim, hidden] — up projection
//! - W2 (down): [hidden, dim] — down projection
//!
//! Forward: out = (SiLU(x @ W1) * (x @ W3)) @ W2
//!
//! ANE implementation uses conv1x1 (3x faster than matmul on Apple Neural Engine).
//! Two-graph split: Graph A (gate+up+SiLU), Graph B (down) — different channel dims.

use ane_bridge::ane::{Graph, Shape};

/// Expert weight set for a single FFN expert.
#[derive(Clone)]
pub struct ExpertWeights {
    /// Gate weights W1: [dim, hidden] row-major.
    pub w1: Vec<f32>,
    /// Up weights W3: [dim, hidden] row-major.
    pub w3: Vec<f32>,
    /// Down weights W2: [hidden, dim] row-major.
    pub w2: Vec<f32>,
    pub dim: usize,
    pub hidden: usize,
}

impl ExpertWeights {
    /// CPU reference SwiGLU expert FFN for a single token.
    /// x: [dim] → out: [dim]
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);

        // Gate: x @ W1 → [hidden]
        let gate = matvec(x, &self.w1, self.dim, self.hidden);

        // Up: x @ W3 → [hidden]
        let up = matvec(x, &self.w3, self.dim, self.hidden);

        // SiLU(gate) * up → [hidden]
        let h: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect();

        // Down: h @ W2 → [dim]
        matvec(&h, &self.w2, self.hidden, self.dim)
    }

    /// CPU reference for batched tokens.
    /// x: [seq, dim] row-major → out: [seq, dim] row-major.
    pub fn forward_batch(&self, x: &[f32], seq: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq * self.dim);
        let mut out = vec![0.0f32; seq * self.dim];
        for s in 0..seq {
            let token = &x[s * self.dim..(s + 1) * self.dim];
            let result = self.forward(token);
            out[s * self.dim..(s + 1) * self.dim].copy_from_slice(&result);
        }
        out
    }

    /// Create random expert weights for testing.
    pub fn random(dim: usize, hidden: usize, seed: u64) -> Self {
        Self {
            w1: pseudo_random(dim * hidden, seed),
            w3: pseudo_random(dim * hidden, seed.wrapping_add(1)),
            w2: pseudo_random(hidden * dim, seed.wrapping_add(2)),
            dim,
            hidden,
        }
    }
}

/// Build ANE conv1x1 graph for gate+up+SiLU projection.
/// Input: [1, dim, 1, seq + 2*hidden]
///   spatial [0:seq] = activations
///   spatial [seq:seq+hidden] = W1 (gate weights)
///   spatial [seq+hidden:seq+2*hidden] = W3 (up weights)
/// Output: [1, hidden, 1, seq]
pub fn build_gate_up_conv(dim: usize, hidden: usize, seq: usize) -> Graph {
    let sp = seq + 2 * hidden;
    let mut g = Graph::new();

    let input = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp });

    // Slice activations: [1, dim, 1, seq]
    let acts = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);

    // Slice W1: [1, dim, 1, hidden] → transpose → reshape → conv1x1
    let w1 = g.slice(input, [0, 0, 0, seq], [1, dim, 1, hidden]);
    let w1_t = g.transpose(w1, [0, 3, 2, 1]); // [1, hidden, 1, dim]
    let w1_conv = g.reshape(w1_t, Shape { batch: hidden, channels: dim, height: 1, width: 1 });
    let gate = g.convolution_2d_1x1_dynamic(acts, w1_conv);

    // Slice W3: [1, dim, 1, hidden] → transpose → reshape → conv1x1
    let w3 = g.slice(input, [0, 0, 0, seq + hidden], [1, dim, 1, hidden]);
    let w3_t = g.transpose(w3, [0, 3, 2, 1]);
    let w3_conv = g.reshape(w3_t, Shape { batch: hidden, channels: dim, height: 1, width: 1 });
    let up = g.convolution_2d_1x1_dynamic(acts, w3_conv);

    // SiLU(gate) * up
    let sig = g.sigmoid(gate);
    let silu = g.multiplication(gate, sig);
    let _gated = g.multiplication(silu, up);

    g
}

/// Build ANE conv1x1 graph for down projection.
/// Input: [1, hidden, 1, seq + dim]
///   spatial [0:seq] = gated hidden activations
///   spatial [seq:seq+dim] = W2 (down weights, stored as [hidden, dim])
/// Output: [1, dim, 1, seq]
pub fn build_down_conv(hidden: usize, dim: usize, seq: usize) -> Graph {
    let sp = seq + dim;
    let mut g = Graph::new();

    let input = g.placeholder(Shape { batch: 1, channels: hidden, height: 1, width: sp });

    let acts = g.slice(input, [0, 0, 0, 0], [1, hidden, 1, seq]);
    let w2 = g.slice(input, [0, 0, 0, seq], [1, hidden, 1, dim]);

    let w2_t = g.transpose(w2, [0, 3, 2, 1]); // [1, dim, 1, hidden]
    let w2_conv = g.reshape(w2_t, Shape { batch: dim, channels: hidden, height: 1, width: 1 });
    let _out = g.convolution_2d_1x1_dynamic(acts, w2_conv);

    g
}

/// Build a simple conv1x1 matmul graph for testing.
/// Input: [1, ic, 1, seq + oc]
/// Output: [1, oc, 1, seq]
pub fn build_conv1x1(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + oc;
    let mut g = Graph::new();

    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });

    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let wts = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);

    let wts_t = g.transpose(wts, [0, 3, 2, 1]);
    let wts_conv = g.reshape(wts_t, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    let _out = g.convolution_2d_1x1_dynamic(acts, wts_conv);

    g
}

/// x @ W where x is [rows] and W is [rows, cols] → [cols]
fn matvec(x: &[f32], w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    for i in 0..rows {
        let xi = x[i];
        let row = &w[i * cols..(i + 1) * cols];
        for j in 0..cols {
            out[j] += xi * row[j];
        }
    }
    out
}

fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX >> 1) as f32) * 0.2 - 0.1
        })
        .collect()
}
