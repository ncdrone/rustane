//! Integration test: ANE conv1x1 expert FFN matches CPU matmul.
//!
//! Verifies that conv1x1 on ANE produces the same output as CPU reference.
//! Tolerance: 1e-2 (ANE computes in fp16 internally).
//!
//! Note: ANE has minimum IOSurface size requirements.
//! Use realistic dims (768 dim, 512+ hidden, seq>=64).

use ane_bridge::ane::{Graph, Shape, TensorData};
use moe_kernels::expert_ffn;
use objc2_foundation::NSQualityOfService;

const TOL: f32 = 1e-2;

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

fn compile_and_eval(graph: Graph, input_data: &[f32], input_shape: Shape, output_shape: Shape) -> Vec<f32> {
    let exe = graph
        .compile(NSQualityOfService::UserInteractive)
        .expect("ANE compilation failed");

    let input_td = TensorData::with_f32(input_data, input_shape);
    let output_td = TensorData::new(output_shape);

    exe.run(&[&input_td], &[&output_td])
        .expect("ANE eval failed");

    output_td.as_f32_slice().to_vec()
}

/// CPU reference matmul on channel-interleaved data.
fn cpu_matmul_ci(acts: &[f32], weights: &[f32], ic: usize, oc: usize, seq: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; oc * seq];
    for s in 0..seq {
        for j in 0..oc {
            let mut sum = 0.0f32;
            for i in 0..ic {
                sum += acts[i * seq + s] * weights[i * oc + j];
            }
            out[j * seq + s] = sum;
        }
    }
    out
}

/// Matmul-based DynMatmul on ANE matches CPU.
#[test]
fn matmul_dyn_matches_cpu() {
    let (ic, oc, seq) = (768, 768, 64);
    let sp = seq + oc;

    let g = build_matmul(ic, oc, seq);
    let input_shape = Shape { batch: 1, channels: ic, height: 1, width: sp };
    let output_shape = Shape { batch: 1, channels: oc, height: 1, width: seq };

    let acts = pseudo_random(ic * seq, 42);
    let weights = pseudo_random(ic * oc, 99);
    let mut input_data = vec![0.0f32; ic * sp];
    for c in 0..ic {
        for s in 0..seq {
            input_data[c * sp + s] = acts[c * seq + s];
        }
        for j in 0..oc {
            input_data[c * sp + seq + j] = weights[c * oc + j];
        }
    }

    let ane_output = compile_and_eval(g, &input_data, input_shape, output_shape);
    let cpu_output = cpu_matmul_ci(&acts, &weights, ic, oc, seq);

    let max_diff = ane_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < TOL,
        "matmul ANE vs CPU: max_diff={max_diff} (expected < {TOL})"
    );
}

/// Conv1x1 on ANE matches CPU matmul.
#[test]
fn conv1x1_matches_cpu() {
    let (ic, oc, seq) = (768, 768, 64);
    let sp = seq + oc;

    let g = expert_ffn::build_conv1x1(ic, oc, seq);
    let input_shape = Shape { batch: 1, channels: ic, height: 1, width: sp };
    let output_shape = Shape { batch: 1, channels: oc, height: 1, width: seq };

    let acts = pseudo_random(ic * seq, 42);
    let weights = pseudo_random(ic * oc, 99);
    let mut input_data = vec![0.0f32; ic * sp];
    for c in 0..ic {
        for s in 0..seq {
            input_data[c * sp + s] = acts[c * seq + s];
        }
        for j in 0..oc {
            input_data[c * sp + seq + j] = weights[c * oc + j];
        }
    }

    let ane_output = compile_and_eval(g, &input_data, input_shape, output_shape);
    let cpu_output = cpu_matmul_ci(&acts, &weights, ic, oc, seq);

    let max_diff = ane_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < TOL,
        "conv1x1 ANE vs CPU: max_diff={max_diff} (expected < {TOL})"
    );
}

/// Full expert FFN on ANE (gate+up+SiLU + down via conv1x1) matches CPU.
#[test]
fn expert_ffn_ane_matches_cpu() {
    let (dim, hidden, seq) = (768, 2048, 64);

    let expert = moe_kernels::ExpertWeights::random(dim, hidden, 42);
    let acts = pseudo_random(dim * seq, 77);

    // CPU reference
    let mut cpu_result = vec![0.0f32; dim * seq];
    for s in 0..seq {
        let x: Vec<f32> = (0..dim).map(|c| acts[c * seq + s]).collect();
        let y = expert.forward(&x);
        for c in 0..dim {
            cpu_result[c * seq + s] = y[c];
        }
    }

    // ANE gate_up
    let sp_gu = seq + 2 * hidden;
    let mut gu_data = vec![0.0f32; dim * sp_gu];
    for c in 0..dim {
        for s in 0..seq {
            gu_data[c * sp_gu + s] = acts[c * seq + s];
        }
        for j in 0..hidden {
            gu_data[c * sp_gu + seq + j] = expert.w1[c * hidden + j];
        }
        for j in 0..hidden {
            gu_data[c * sp_gu + seq + hidden + j] = expert.w3[c * hidden + j];
        }
    }

    let gu_in_shape = Shape { batch: 1, channels: dim, height: 1, width: sp_gu };
    let gu_out_shape = Shape { batch: 1, channels: hidden, height: 1, width: seq };
    let gated = compile_and_eval(
        expert_ffn::build_gate_up_conv(dim, hidden, seq),
        &gu_data,
        gu_in_shape,
        gu_out_shape,
    );

    // ANE down
    let sp_down = seq + dim;
    let mut down_data = vec![0.0f32; hidden * sp_down];
    for c in 0..hidden {
        for s in 0..seq {
            down_data[c * sp_down + s] = gated[c * seq + s];
        }
        for j in 0..dim {
            down_data[c * sp_down + seq + j] = expert.w2[c * dim + j];
        }
    }

    let down_in_shape = Shape { batch: 1, channels: hidden, height: 1, width: sp_down };
    let down_out_shape = Shape { batch: 1, channels: dim, height: 1, width: seq };
    let ane_result = compile_and_eval(
        expert_ffn::build_down_conv(hidden, dim, seq),
        &down_data,
        down_in_shape,
        down_out_shape,
    );

    let max_diff = ane_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Accumulated fp16 error across 3 projections + SiLU
    let ffn_tol = 0.1;
    assert!(
        max_diff < ffn_tol,
        "expert FFN ANE vs CPU: max_diff={max_diff} (expected < {ffn_tol})"
    );
}

fn build_matmul(ic: usize, oc: usize, seq: usize) -> Graph {
    let sp = seq + oc;
    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: sp });
    let acts = g.slice(input, [0, 0, 0, 0], [1, ic, 1, seq]);
    let wts = g.slice(input, [0, 0, 0, seq], [1, ic, 1, oc]);
    let acts_r = g.reshape(acts, Shape { batch: 1, channels: 1, height: ic, width: seq });
    let acts_t = g.transpose(acts_r, [0, 1, 3, 2]);
    let wts_r = g.reshape(wts, Shape { batch: 1, channels: 1, height: ic, width: oc });
    let mm = g.matrix_multiplication(acts_t, wts_r, false, false);
    let mm_t = g.transpose(mm, [0, 1, 3, 2]);
    let _out = g.reshape(mm_t, Shape { batch: 1, channels: oc, height: 1, width: seq });
    g
}
