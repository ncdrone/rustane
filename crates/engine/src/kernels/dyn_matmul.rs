//! DynMatmul — the core kernel pattern for dynamic weight training on ANE.
//!
//! Input IOSurface: [1, IC, 1, SEQ + OC]  (activations + weights packed in spatial dim)
//! Output: [1, OC, 1, SEQ]
//!
//! Graph: slice acts → slice weights → reshape → transpose → matmul → transpose → reshape
//!
//! Used by: woFwd, ffnBwdW2t, wotBwd, qBwd

use ane_bridge::ane::{Graph, Shape};

/// Build a single DynMatmul graph: y = acts @ weights
/// Input: [1, ic, 1, seq + oc]
/// Output: [1, oc, 1, seq]
pub fn build(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + oc;
    let mut g = Graph::new();

    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });

    // Slice activations: [1, IC, 1, SEQ] at offset 0
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);

    // Slice weights: [1, IC, 1, OC] at offset SEQ
    let wts = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);

    // Reshape acts: [1, IC, 1, SEQ] → [1, 1, IC, SEQ]
    let acts_r = g.reshape(acts, Shape { batch: 1, channels: 1, height: ic, width: seq });

    // Transpose acts: [1, 1, IC, SEQ] → [1, 1, SEQ, IC]
    let acts_t = g.transpose(acts_r, [0, 1, 3, 2]);

    // Reshape weights: [1, IC, 1, OC] → [1, 1, IC, OC]
    let wts_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: ic, width: oc });

    // Matmul: [1, 1, SEQ, IC] @ [1, 1, IC, OC] → [1, 1, SEQ, OC]
    let mm = g.matrix_multiplication(acts_t, wts_r, false, false);

    // Transpose back: [1, 1, SEQ, OC] → [1, 1, OC, SEQ]
    let mm_t = g.transpose(mm, [0, 1, 3, 2]);

    // Reshape to output format: [1, 1, OC, SEQ] → [1, OC, 1, SEQ]
    let _out = g.reshape(mm_t, Shape { batch: 1, channels: oc, height: 1, width: seq });

    g
}

/// Build a DualDynMatmul graph: two parallel matmuls from one input.
/// Input: [1, IC, 1, 2*SEQ + 2*OC]
///   acts1 at [0..SEQ], wts1 at [SEQ..SEQ+OC], acts2 at [SEQ+OC..2*SEQ+OC], wts2 at [2*SEQ+OC..]
/// Output: [1, OC, 1, SEQ] (sum of both matmuls)
///
/// Used by: ffnBwdW13t, kvBwd
pub fn build_dual(ic: usize, oc: usize, seq: usize) -> Graph {
    let mut g = Graph::new();
    let sp = 2 * seq + 2 * oc;

    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });

    // First matmul: acts1 @ wts1
    let acts1 = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let wts1 = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let acts1_r = g.reshape(acts1, Shape { batch: 1, channels: 1, height: ic, width: seq });
    let acts1_t = g.transpose(acts1_r, [0, 1, 3, 2]);
    let wts1_r = g.reshape(wts1, Shape { batch: 1, channels: 1, height: ic, width: oc });
    let mm1 = g.matrix_multiplication(acts1_t, wts1_r, false, false);
    let mm1_t = g.transpose(mm1, [0, 1, 3, 2]);
    let out1 = g.reshape(mm1_t, Shape { batch: 1, channels: oc, height: 1, width: seq });

    // Second matmul: acts2 @ wts2
    let off2_acts = seq + oc;
    let off2_wts = 2 * seq + oc;
    let acts2 = g.slice(input, [0, 0, 0, off2_acts], [1, ic, 1, seq]);
    let wts2 = g.slice(input, [0, 0, 0, off2_wts], [1, ic, 1, oc]);
    let acts2_r = g.reshape(acts2, Shape { batch: 1, channels: 1, height: ic, width: seq });
    let acts2_t = g.transpose(acts2_r, [0, 1, 3, 2]);
    let wts2_r = g.reshape(wts2, Shape { batch: 1, channels: 1, height: ic, width: oc });
    let mm2 = g.matrix_multiplication(acts2_t, wts2_r, false, false);
    let mm2_t = g.transpose(mm2, [0, 1, 3, 2]);
    let out2 = g.reshape(mm2_t, Shape { batch: 1, channels: oc, height: 1, width: seq });

    // Sum both paths
    let _sum = g.addition(out1, out2);

    g
}

/// Build a single DynMatmul graph using conv1x1: y = conv1x1(acts, weights)
/// Input: [1, ic, 1, seq + oc]
/// Output: [1, oc, 1, seq]
///
/// Uses 4 ops (slice, slice, reshape, conv1x1) instead of 7 ops (matmul path).
/// ANE is a convolution engine — conv1x1 should be 3x faster per maderix benchmarks.
pub fn build_conv(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + oc;
    let mut g = Graph::new();

    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });

    // Slice activations: [1, IC, 1, SEQ]
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);

    // Slice weights: [1, IC, 1, OC]
    let wts = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);

    // Transpose weights: [1, IC, 1, OC] → [1, OC, 1, IC] (swap channels↔width)
    // This puts weights in [OC, IC] row-major order needed by conv
    let wts_t = g.transpose(wts, [0, 3, 2, 1]);

    // Reshape to conv weight format: [1, OC, 1, IC] → [OC, IC, 1, 1]
    let wts_conv = g.reshape(wts_t, Shape { batch: oc, channels: ic, height: 1, width: 1 });

    // Conv1x1: [1, IC, 1, SEQ] * [OC, IC, 1, 1] → [1, OC, 1, SEQ]
    let _out = g.convolution_2d_1x1_dynamic(acts, wts_conv);

    g
}

/// Build a DualDynMatmul graph using conv1x1: two parallel conv1x1 ops summed.
/// Input: [1, IC, 1, 2*SEQ + 2*OC]
/// Output: [1, OC, 1, SEQ] (sum of both conv1x1 results)
pub fn build_dual_conv(ic: usize, oc: usize, seq: usize) -> Graph {
    let mut g = Graph::new();
    let sp = 2 * seq + 2 * oc;

    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });

    // First conv1x1: acts1 * wts1
    let acts1 = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let wts1 = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let wts1_t = g.transpose(wts1, [0, 3, 2, 1]);
    let wts1_conv = g.reshape(wts1_t, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    let out1 = g.convolution_2d_1x1_dynamic(acts1, wts1_conv);

    // Second conv1x1: acts2 * wts2
    let off2_acts = seq + oc;
    let off2_wts = 2 * seq + oc;
    let acts2 = g.slice(input, [0, 0, 0, off2_acts], [1, ic, 1, seq]);
    let wts2 = g.slice(input, [0, 0, 0, off2_wts], [1, ic, 1, oc]);
    let wts2_t = g.transpose(wts2, [0, 3, 2, 1]);
    let wts2_conv = g.reshape(wts2_t, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    let out2 = g.convolution_2d_1x1_dynamic(acts2, wts2_conv);

    // Sum both paths
    let _sum = g.addition(out1, out2);

    g
}

/// IOSurface spatial width for a single DynMatmul kernel.
pub fn spatial_width(seq: usize, oc: usize) -> usize {
    seq + oc
}

/// IOSurface spatial width for a DualDynMatmul kernel.
pub fn dual_spatial_width(seq: usize, oc: usize) -> usize {
    2 * seq + 2 * oc
}
