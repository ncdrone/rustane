//! ANE prefill attention vs CPU reference.
//!
//! Verifies that the fused ANE GQA graph produces the same output as the
//! CPU batched attention implementation within fp16 tolerance.

use ane_bridge::ane::{Shape, TensorData};
use moe_kernels::gqa_prefill;
use moe_infer::attention::{GqaConfig, RopeTables, gqa_forward_batch_f32};
use objc2_foundation::NSQualityOfService;

const HIDDEN: usize = 2048;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM; // 4096
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 512
const ROPE_THETA: f32 = 1_000_000.0;
const EPS: f32 = 1e-6;
const TOL: f32 = 0.05;

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

/// Random values near 1.0 (realistic for RMSNorm scale weights).
fn random_norm_weights(n: usize, seed: u64) -> Vec<f32> {
    pseudo_random(n, seed).iter().map(|&v| 1.0 + v * 2.0).collect()
}

fn run_ane_prefill_test(seq: usize) {
    let cfg = GqaConfig {
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        max_seq: 4096,
    };
    let rope = RopeTables::build(seq, HEAD_DIM, ROPE_THETA);

    // Random data
    let xs = pseudo_random(seq * HIDDEN, 42);
    let q_proj = pseudo_random(Q_DIM * HIDDEN, 100);
    let k_proj = pseudo_random(KV_DIM * HIDDEN, 200);
    let v_proj = pseudo_random(KV_DIM * HIDDEN, 300);
    let q_norm_w = random_norm_weights(HEAD_DIM, 400);
    let k_norm_w = random_norm_weights(HEAD_DIM, 500);

    // ── CPU reference ──
    let cpu = gqa_forward_batch_f32(
        &xs, seq, &q_proj, &k_proj, &v_proj,
        &q_norm_w, &k_norm_w, &rope, &cfg, EPS,
    );

    // ── ANE graph (two inputs: acts + weights) ──
    let graph = gqa_prefill::build_gqa_prefill(
        HIDDEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        seq, ROPE_THETA, EPS, &q_norm_w, &k_norm_w,
    );

    let wts_sp = gqa_prefill::weight_spatial_width(Q_DIM, KV_DIM);
    let out_ch = gqa_prefill::output_channels(Q_DIM, KV_DIM);

    // ── Stage activations into IOSurface 0 ──
    let mut acts_data = vec![0.0f32; HIDDEN * seq];
    for c in 0..HIDDEN {
        for t in 0..seq {
            acts_data[c * seq + t] = xs[t * HIDDEN + c];
        }
    }

    // ── Stage weights into IOSurface 1 (transposed, done once) ──
    let mut wts_data = vec![0.0f32; HIDDEN * wts_sp];
    for c in 0..HIDDEN {
        for i in 0..Q_DIM {
            wts_data[c * wts_sp + i] = q_proj[i * HIDDEN + c];
        }
        for i in 0..KV_DIM {
            wts_data[c * wts_sp + Q_DIM + i] = k_proj[i * HIDDEN + c];
        }
        for i in 0..KV_DIM {
            wts_data[c * wts_sp + Q_DIM + KV_DIM + i] = v_proj[i * HIDDEN + c];
        }
    }

    // ── Compile and run ──
    let acts_shape = Shape { batch: 1, channels: HIDDEN, height: 1, width: seq };
    let wts_shape = Shape { batch: 1, channels: HIDDEN, height: 1, width: wts_sp };
    let output_shape = Shape { batch: 1, channels: out_ch, height: 1, width: seq };

    let exe = graph
        .compile(NSQualityOfService::UserInteractive)
        .expect("ANE compilation failed");

    let acts_td = TensorData::with_f32(&acts_data, acts_shape);
    let wts_td = TensorData::with_f32(&wts_data, wts_shape);
    let output_td = TensorData::new(output_shape);

    exe.run(&[&acts_td, &wts_td], &[&output_td])
        .expect("ANE dispatch failed");

    let output_data = output_td.as_f32_slice().to_vec();

    // ── Read ANE output (channel-interleaved → row-major) ──
    let mut ane_attn = vec![0.0f32; seq * Q_DIM];
    let mut ane_k = vec![0.0f32; seq * KV_DIM];
    let mut ane_v = vec![0.0f32; seq * KV_DIM];
    for t in 0..seq {
        for i in 0..Q_DIM {
            ane_attn[t * Q_DIM + i] = output_data[i * seq + t];
        }
        for i in 0..KV_DIM {
            ane_k[t * KV_DIM + i] = output_data[(Q_DIM + i) * seq + t];
        }
        for i in 0..KV_DIM {
            ane_v[t * KV_DIM + i] = output_data[(Q_DIM + KV_DIM + i) * seq + t];
        }
    }

    // ── Compare ──
    let diff_attn = max_diff(&ane_attn, &cpu.attn_out);
    let diff_k = max_diff(&ane_k, &cpu.k_rope);
    let diff_v = max_diff(&ane_v, &cpu.v);

    eprintln!("seq={seq}: attn max_diff={diff_attn:.6}, k max_diff={diff_k:.6}, v max_diff={diff_v:.6}");

    assert!(
        diff_v < TOL,
        "V mismatch at seq={seq}: max_diff={diff_v} (expected < {TOL})"
    );
    assert!(
        diff_k < TOL,
        "K_rope mismatch at seq={seq}: max_diff={diff_k} (expected < {TOL})"
    );
    assert!(
        diff_attn < TOL,
        "attn_out mismatch at seq={seq}: max_diff={diff_attn} (expected < {TOL})"
    );
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn ane_gqa_matches_cpu_seq64() {
    run_ane_prefill_test(64);
}

#[test]
fn ane_gqa_matches_cpu_seq128() {
    run_ane_prefill_test(128);
}
